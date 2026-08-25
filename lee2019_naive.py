"""
Naive deletion on Lee2019.
"""

import json
import time
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

CLF_NAMES = ["RandomForest", "ExtraTrees", "SGDClassifier", "RidgeClassifier"]


def load_data():
    psd      = np.load("data/X_lee2019_scaled_log_psd_nooverlap.npy")
    stat     = np.load("data/X_lee2019_statistical_nooverlap.npy")
    features = np.concatenate([psd, stat], axis=1)
    labels   = np.load("data/y_lee2019_nooverlap.npy")
    groups   = np.load("data/runs_lee2019_nooverlap.npy")
    return features, labels, groups


def make_classifier(clf_name):
    if clf_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1)
    elif clf_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1)
    elif clf_name == "SGDClassifier":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(
                loss="modified_huber", max_iter=1000,
                random_state=42, n_jobs=-1, tol=1e-3))
        ])
    elif clf_name == "RidgeClassifier":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(alpha=1.0))
        ])


def get_classes(clf, clf_name):
    if clf_name in ("RandomForest", "ExtraTrees"):
        return clf.classes_
    return clf.named_steps["clf"].classes_


def get_proba_and_preds(clf, clf_name, X):
    if clf_name in ("RandomForest", "ExtraTrees", "SGDClassifier"):
        return clf.predict_proba(X), clf.predict(X)
    elif clf_name == "RidgeClassifier":
        decision = clf.decision_function(X)
        exp_d    = np.exp(decision - decision.max(axis=1, keepdims=True))
        proba    = exp_d / exp_d.sum(axis=1, keepdims=True)
        return proba, clf.predict(X)


def compute_mia_auc(member_proba, nonmember_proba):
    """Orientation-invariant separability of R03 and R04 windows."""
    if len(member_proba) == 0 or len(nonmember_proba) == 0:
        return 0.5
    member_entropy    = scipy_entropy(member_proba.T)
    nonmember_entropy = scipy_entropy(nonmember_proba.T)
    scores = np.concatenate([member_entropy, nonmember_entropy])
    mia_labels = np.concatenate([
        np.ones(len(member_entropy)),
        np.zeros(len(nonmember_entropy))
    ])
    try:
        auc = float(roc_auc_score(mia_labels, scores))
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def analyze_target_subject(clf, clf_name, X_target, target_subject):
    proba, preds = get_proba_and_preds(clf, clf_name, X_target)
    classes_list = list(get_classes(clf, clf_name))

    class_counts   = {c: int(np.sum(preds == c)) for c in classes_list}
    top_class      = max(class_counts, key=class_counts.get)
    top_class_frac = class_counts[top_class] / len(preds)

    mean_proba         = proba.mean(axis=0)
    pred_entropy       = float(scipy_entropy(mean_proba))
    max_entropy        = float(np.log(len(classes_list)))
    normalized_entropy = pred_entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "target_subject":                target_subject,
        "n_test_windows":                int(len(X_target)),
        "top_predicted_class":           top_class,
        "prediction_entropy_normalized": float(normalized_entropy),
        "consistency":                   float(top_class_frac),
        "correctly_identified":          bool(top_class == target_subject),
    }


if __name__ == "__main__":
    t_start = time.time()

    features, labels, groups = load_data()
    subjects = sorted(np.unique(labels).tolist())
    n        = len(subjects)

    print("Method: naive deletion")
    print("Dataset: Lee2019")
    print(f"Subjects: {n}, Features: {features.shape[1]}")
    print(f"Windows: {features.shape[0]}")

    r03_mask = np.char.find(groups, "R03") != -1
    r04_mask = np.char.find(groups, "R04") != -1

    print("\nTraining original models")
    X_tr = features[r03_mask]
    y_tr = labels[r03_mask]
    X_te = features[r04_mask]
    y_te = labels[r04_mask]

    original_models  = {}
    baseline_results = {}

    for clf_name in CLF_NAMES:
        clf = make_classifier(clf_name)
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        baseline_results[clf_name] = float(acc)
        original_models[clf_name]  = clf
        print(f"  {clf_name}: {acc:.4f}")
    print(f"Training done ({time.time() - t_start:.1f}s elapsed)")

    r03_windows_per_subject = {}
    for s in subjects:
        idx = np.where(r03_mask & (labels == s))[0]
        r03_windows_per_subject[s] = features[idx]

    all_results = {clf_name: [] for clf_name in CLF_NAMES}

    print("\nScoring all subjects against the original model")

    for t_idx, target in enumerate(subjects):
        X_target_r04 = features[r04_mask & (labels == target)]
        if len(X_target_r04) == 0:
            continue

        X_target_r03 = r03_windows_per_subject[target]

        for clf_name in CLF_NAMES:
            clf = original_models[clf_name]

            metrics = analyze_target_subject(
                clf, clf_name, X_target_r04, target)

            member_proba, _    = get_proba_and_preds(
                clf, clf_name, X_target_r03)
            nonmember_proba, _ = get_proba_and_preds(
                clf, clf_name, X_target_r04)

            result = {
                **metrics,
                "mia_auc":    compute_mia_auc(member_proba, nonmember_proba),
                "classifier": clf_name,
            }
            all_results[clf_name].append(result)

        if t_idx < 5 or (t_idx + 1) % 10 == 0:
            print(f"  [{t_idx + 1}/{n}] {target} done "
                  f"({time.time() - t_start:.1f}s elapsed)")

    print("\n" + "=" * 70)
    print("NAIVE DELETION SUMMARY (Lee2019)")
    print("=" * 70)

    print("\nBaseline accuracy:")
    for clf_name, acc in baseline_results.items():
        print(f"  {clf_name}: {acc:.4f}")

    print(f"\n{'Classifier':<22} {'Consistency':>13} "
          f"{'Entropy':>14} {'MIA AUC':>10} {'Correct ID%':>12}")
    print("-" * 75)

    for clf_name in CLF_NAMES:
        res = all_results[clf_name]
        if not res:
            continue
        mc  = np.mean([r["consistency"] for r in res])
        me  = np.mean([r["prediction_entropy_normalized"] for r in res])
        mia = np.mean([r["mia_auc"] for r in res])
        cid = np.mean([r["correctly_identified"] for r in res])
        print(f"{clf_name:<22} {mc:>13.4f} {me:>14.4f} "
              f"{mia:>10.4f} {cid:>11.1%}")

    with open("summary_naive_lee2019.json", "w") as f:
        json.dump({
            "method":            "naive_deletion",
            "dataset":           "Lee2019",
            "baseline_accuracy": baseline_results,
            "per_target":        all_results,
        }, f, indent=2)

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    print("Saved summary_naive_lee2019.json")