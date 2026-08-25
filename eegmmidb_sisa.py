import json
import time
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

CLF_NAMES = ["RandomForest", "ExtraTrees", "SGDClassifier", "RidgeClassifier"]
N_SHARDS  = 4


def load_data():
    features = np.load("data/features_spectral_nonoverlap.npy")
    labels   = np.load("data/labels_spectral_nonoverlap.npy")
    groups   = np.load("data/groups_spectral_nonoverlap.npy")
    return features, labels, groups


def make_classifier(clf_name):
    """Tree ensembles use 50 estimators per shard, giving 200 in total."""
    if clf_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=50, random_state=42, n_jobs=-1)
    elif clf_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=50, random_state=42, n_jobs=-1)
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


def get_proba(clf, clf_name, X, all_classes):
    """Predict and align shard probabilities to the full class vector."""
    if clf_name in ("RandomForest", "ExtraTrees", "SGDClassifier"):
        proba_raw = clf.predict_proba(X)
    elif clf_name == "RidgeClassifier":
        decision  = clf.decision_function(X)
        exp_d     = np.exp(decision - decision.max(axis=1, keepdims=True))
        proba_raw = exp_d / exp_d.sum(axis=1, keepdims=True)

    clf_classes   = list(get_classes(clf, clf_name))
    proba_aligned = np.zeros((len(X), len(all_classes)))
    for j, c in enumerate(clf_classes):
        if c in all_classes:
            proba_aligned[:, all_classes.index(c)] = proba_raw[:, j]
    return proba_aligned


def aggregate_proba(models, clf_name, X, all_classes):
    proba_sum = np.zeros((len(X), len(all_classes)))
    for model in models:
        proba_sum += get_proba(model, clf_name, X, all_classes)
    return proba_sum / len(models)


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


def analyze_target_subject(proba, all_classes):
    classes_list = list(all_classes)
    preds        = np.array([classes_list[np.argmax(p)] for p in proba])

    class_counts   = {c: int(np.sum(preds == c)) for c in classes_list}
    top_class      = max(class_counts, key=class_counts.get)
    top_class_frac = class_counts[top_class] / len(preds)

    mean_proba         = proba.mean(axis=0)
    pred_entropy       = float(scipy_entropy(mean_proba))
    max_entropy        = float(np.log(len(classes_list)))
    normalized_entropy = pred_entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "top_predicted_class":           top_class,
        "prediction_entropy_normalized": float(normalized_entropy),
        "consistency":                   float(top_class_frac),
    }


if __name__ == "__main__":
    t_start = time.time()

    features, labels, groups = load_data()
    subjects    = sorted(np.unique(labels).tolist())
    n           = len(subjects)
    all_classes = subjects

    print(f"Method: SISA (k={N_SHARDS} shards, subject-level)")
    print("Dataset: EEGMMIDB")
    print(f"Subjects: {n}, Features: {features.shape[1]}")
    print(f"Windows: {features.shape[0]}")

    r03_mask = np.char.find(groups, "R03") != -1
    r04_mask = np.char.find(groups, "R04") != -1
    r03_idx  = np.where(r03_mask)[0]
    r04_idx  = np.where(r04_mask)[0]

    rng = np.random.default_rng(42)
    subject_to_shard = {
        s: i % N_SHARDS
        for i, s in enumerate(rng.permutation(subjects))
    }

    print("\nShard assignment:")
    for shard_id in range(N_SHARDS):
        count = sum(1 for sh in subject_to_shard.values() if sh == shard_id)
        print(f"  Shard {shard_id}: {count} subjects")

    r03_windows_per_subject = {}
    for s in subjects:
        idx = r03_idx[labels[r03_idx] == s]
        r03_windows_per_subject[s] = features[idx]

    all_results       = {clf_name: [] for clf_name in CLF_NAMES}
    completed_targets = set()

    try:
        with open("summary_sisa_eegmmidb_partial.json") as f:
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

    print("\nRunning per-target shard retraining")

    for t_idx, target in enumerate(subjects):
        if target in completed_targets:
            continue

        target_shard = subject_to_shard[target]
        X_target_r04 = features[r04_idx[labels[r04_idx] == target]]
        if len(X_target_r04) == 0:
            continue

        X_target_r03 = r03_windows_per_subject[target]

        for clf_name in CLF_NAMES:
            shard_models = []
            for shard_id in range(N_SHARDS):
                shard_subjects = [s for s, sh in subject_to_shard.items()
                                  if sh == shard_id]
                if shard_id == target_shard:
                    shard_subjects = [s for s in shard_subjects if s != target]

                if len(shard_subjects) < 2:
                    continue

                shard_r03_idx = [i for i in r03_idx
                                 if labels[i] in shard_subjects]
                if len(shard_r03_idx) == 0:
                    continue

                X_shard = features[shard_r03_idx]
                y_shard = labels[shard_r03_idx]
                if len(np.unique(y_shard)) < 2:
                    continue

                clf = make_classifier(clf_name)
                clf.fit(X_shard, y_shard)
                shard_models.append(clf)

            if not shard_models:
                continue

            proba_agg = aggregate_proba(
                shard_models, clf_name, X_target_r04, all_classes)
            metrics = analyze_target_subject(proba_agg, all_classes)

            member_proba = aggregate_proba(
                shard_models, clf_name, X_target_r03, all_classes)

            result = {
                "target_subject": target,
                "n_test_windows": int(len(X_target_r04)),
                "classifier":     clf_name,
                "mia_auc":        compute_mia_auc(member_proba, proba_agg),
                **metrics,
            }
            all_results[clf_name].append(result)

        with open("summary_sisa_eegmmidb_partial.json", "w") as f:
            json.dump({
                "method":     "sisa",
                "dataset":    "EEGMMIDB",
                "n_shards":   N_SHARDS,
                "per_target": all_results,
            }, f, indent=2)

        if t_idx < 5 or (t_idx + 1) % 10 == 0:
            print(f"  [{t_idx + 1}/{n}] {target} done "
                  f"({time.time() - t_start:.1f}s elapsed)")

    print("\n" + "=" * 70)
    print(f"SISA SUMMARY (k={N_SHARDS}, EEGMMIDB)")
    print("=" * 70)

    print(f"\n{'Classifier':<22} {'Consistency':>13} "
          f"{'Entropy':>14} {'MIA AUC':>10}")
    print("-" * 62)

    for clf_name in CLF_NAMES:
        res = all_results[clf_name]
        if not res:
            continue
        mc  = np.mean([r["consistency"] for r in res])
        me  = np.mean([r["prediction_entropy_normalized"] for r in res])
        mia = np.mean([r["mia_auc"] for r in res])
        print(f"{clf_name:<22} {mc:>13.4f} {me:>14.4f} {mia:>10.4f}")

    with open("summary_sisa_eegmmidb.json", "w") as f:
        json.dump({
            "method":     "sisa",
            "dataset":    "EEGMMIDB",
            "n_shards":   N_SHARDS,
            "per_target": all_results,
        }, f, indent=2)

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    print("Saved summary_sisa_eegmmidb.json")