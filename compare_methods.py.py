"""
Comparison tables across the five evaluated methods. 
"""

import json
import numpy as np

CLF_NAMES = ["RandomForest", "ExtraTrees", "SGDClassifier", "RidgeClassifier"]
CLF_SHORT = ["RF", "ET", "SGD", "Ridge"]

METHODS = ["Naive Deletion", "TGP", "RLP", "SISA", "Exact Retraining"]

DATASETS = {
    "EEGMMIDB": {
        "Naive Deletion":   "summary_naive_eegmmidb.json",
        "TGP":              "summary_tgp_eegmmidb.json",
        "RLP":              "summary_rlp_eegmmidb.json",
        "SISA":             "summary_sisa_eegmmidb.json",
        "Exact Retraining": "summary_exact_eegmmidb.json",
    },
    "Lee2019": {
        "Naive Deletion":   "summary_naive_lee2019.json",
        "TGP":              "summary_tgp_lee2019.json",
        "RLP":              "summary_rlp_lee2019.json",
        "SISA":             "summary_sisa_lee2019.json",
        "Exact Retraining": "summary_exact_lee2019.json",
    },
}

METRICS = {
    "entropy":     ("prediction_entropy_normalized",
                    "AGGREGATE ENTROPY (higher = better forgetting)"),
    "consistency": ("consistency",
                    "CONSISTENCY (lower = better forgetting)"),
    "mia":         ("mia_auc",
                    "MIA AUC (0.5 = low separability)"),
}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def mean_metric(data, clf_name, key):
    """Mean of one metric over all target subjects, or None if unavailable."""
    if data is None:
        return None
    results = data.get("per_target", {}).get(clf_name, [])
    values  = [r[key] for r in results if key in r]
    return float(np.mean(values)) if values else None


def print_table(dataset, summaries, key, title):
    print("\n" + "=" * 62)
    print(f"{dataset} — {title}")
    print("=" * 62)
    print(f"{'Method':<20}" + "".join(f"{s:>10}" for s in CLF_SHORT))
    print("-" * 62)
    for method in METHODS:
        row = f"{method:<20}"
        for clf in CLF_NAMES:
            value = mean_metric(summaries[method], clf, key)
            row += f"{value:>10.3f}" if value is not None else f"{'N/A':>10}"
        print(row)


def subject_count(summaries):
    """Number of target subjects present, for a sanity check."""
    for data in summaries.values():
        if data is not None:
            results = data.get("per_target", {}).get(CLF_NAMES[0], [])
            if results:
                return len(results)
    return 0


if __name__ == "__main__":
    combined = {}

    for dataset, files in DATASETS.items():
        summaries = {m: load(p) for m, p in files.items()}
        missing   = [m for m, d in summaries.items() if d is None]
        if missing:
            print(f"\n{dataset}: missing results for {', '.join(missing)}")

        print(f"\n\n{'#' * 62}")
        print(f"# {dataset}  (n = {subject_count(summaries)} subjects)")
        print(f"{'#' * 62}")

        for key, (field, title) in METRICS.items():
            print_table(dataset, summaries, field, title)

        combined[dataset] = {
            method: {
                clf: {
                    key: mean_metric(summaries[method], clf, field)
                    for key, (field, _) in METRICS.items()
                }
                for clf in CLF_NAMES
            }
            for method in METHODS
        }

    with open("summary_comparison.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\nSaved summary_comparison.json")