"""
Targeted Gaussian Perturbation (TGP) on Lee2019.
Inputs: data/X_lee2019_scaled_log_psd_nooverlap.npy
        data/X_lee2019_statistical_nooverlap.npy
Output: summary_tgp_lee2019.json
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

CLF_NAMES   = ["RandomForest", "ExtraTrees", "SGDClassifier", "RidgeClassifier"]
NOISE_SCALE = 0.5


def load_data():
    psd      = np.load("data/X_lee2019_scaled_log_psd_nooverlap.npy")
    stat     = np.load("data/X_lee2019_statistical_nooverlap.npy")
    features = np.concatenate([psd, stat], axis=1)
    labels   = np.load("data/y_lee2019_nooverlap.npy")
    groups   = np.load("data/runs_lee2019_nooverlap.npy")
    return features, labels, groups


def get_train_test_perturbed(features, labels, groups,
                             target_subject, noise_scale, target_idx):
    """Perturb the target subject's R03 windows, keeping their label."""
    rng      = np.random.default_rng(target_idx)
    r03_mask = np.char.find(groups, "R03") != -1
    r04_mask = np.char.find(groups, "R04") != -1

    X_train = features[r03_mask].copy()
    y_train = labels[r03_mask].copy()
    X_test  = features[r04_mask]
    y_test  = labels[r04_mask]

    target_mask = y_train == target_subject
    feature_std = features[r03_mask].std(axis=0)
    X_train[target_mask] += rng.normal(
        0, noise_scale * feature_std,
        size=X_train[target_mask].shape)

    return X_train, y_train, X_test, y_test


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

    print(f"Method: Targeted Gaussian Perturbation (noise scale {NOISE_SCALE})")
    print("Dataset: Lee2019")
    print(f"Subjects: {n}, Features: {features.shape[1]}")
    print(f"Windows: {features.shape[0]}")

    r03_mask = np.char.find(groups, "R03") != -1
    r04_mask = np.char.find(groups, "R04") != -1

    print("\nTraining baseline (unperturbed)...")
    X_tr_base = features[r03_mask]
    y_tr_base = labels[r03_mask]
    X_te_base = features[r04_mask]
    y_te_base = labels[r04_mask]

    baseline_results = {}
    for clf_name in CLF_NAMES:
        clf = make_classifier(clf_name)
        clf.fit(X_tr_base, y_tr_base)
        acc = accuracy_score(y_te_base, clf.predict(X_te_base))
        baseline_results[clf_name] = float(acc)
        print(f"  {clf_name}: {acc:.4f}")
    print(f"Baseline done ({time.time() - t_start:.1f}s elapsed)")

    r03_windows_per_subject = {}
    for s in subjects:
        idx = np.where(r03_mask & (labels == s))[0]
        r03_windows_per_subject[s] = features[idx]

    all_results       = {clf_name: [] for clf_name in CLF_NAMES}
    completed_targets = set()

    try:
        with open("summary_tgp_lee2019_partial.json") as f:
            saved = json.load(f)
        all_results = saved.get("per_target", all_results)
        counts = {}
        for clf_name, res_list in all_results.items():
            for r in res_list:
                t = r["target_subject"]
                counts[t] = counts.get(t, 0) + 1
        completed_targets = {t for t, c in counts.items()
                             if c == len(CLF_NAMES)}
        print(f"Resuming: {len(completed_targets)} targets done.")
    except FileNotFoundError:
        pass

    print("\nRunning per-target perturbation and retraining...")

    for t_idx, target in enumerate(subjects):
        if target in completed_targets:
            continue

        X_target_r04 = features[r04_mask & (labels == target)]
        if len(X_target_r04) == 0:
            continue

        X_target_r03 = r03_windows_per_subject[target]

        X_tr, y_tr, _, _ = get_train_test_perturbed(
            features, labels, groups,
            target_subject=target,
            noise_scale=NOISE_SCALE,
            target_idx=t_idx)

        for clf_name in CLF_NAMES:
            clf = make_classifier(clf_name)
            clf.fit(X_tr, y_tr)

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

        with open("summary_tgp_lee2019_partial.json", "w") as f:
            json.dump({
                "method":            "targeted_gaussian_perturbation",
                "dataset":           "Lee2019",
                "noise_scale":       NOISE_SCALE,
                "baseline_accuracy": baseline_results,
                "per_target":        all_results,
            }, f, indent=2)

        if t_idx < 5 or (t_idx + 1) % 10 == 0:
            print(f"  [{t_idx + 1}/{n}] {target} done "
                  f"({time.time() - t_start:.1f}s elapsed)")

    print("\n" + "=" * 70)
    print("TARGETED GAUSSIAN PERTURBATION SUMMARY (Lee2019)")
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

    with open("summary_tgp_lee2019.json", "w") as f:
        json.dump({
            "method":            "targeted_gaussian_perturbation",
            "dataset":           "Lee2019",
            "noise_scale":       NOISE_SCALE,
            "baseline_accuracy": baseline_results,
            "per_target":        all_results,
        }, f, indent=2)

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    print("Saved summary_tgp_lee2019.json")