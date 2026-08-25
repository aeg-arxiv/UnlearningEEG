"""
Nearest-neighbour baseline check on EEGMMIDB.

Tests whether post-deletion assignments reflect raw EEG similarity rather than
learned representations. For each target subject the pre-deletion nearest
neighbour is found among all other subjects, using Euclidean distance over
mean R03 feature vectors, and compared against the dominant post-unlearning
predicted class under exact retraining.

A low match rate indicates that residual assignments are not driven by feature
proximity.

Inputs: data/features_spectral_nonoverlap.npy
        summary_exact_eegmmidb.json
Output: summary_nn_baseline.json
"""

import json
import numpy as np
from scipy.spatial.distance import cdist

CLF_NAMES = ["RandomForest", "ExtraTrees", "SGDClassifier", "RidgeClassifier"]


def load_data():
    features = np.load("data/features_spectral_nonoverlap.npy")
    labels   = np.load("data/labels_spectral_nonoverlap.npy")
    groups   = np.load("data/groups_spectral_nonoverlap.npy")
    return features, labels, groups


def build_nearest_neighbour_map(features, labels, groups, subjects):
    """Map each subject to their closest other subject"""
    r03_mask = np.char.find(groups, "R03") != -1

    centroids = np.array([
        features[r03_mask & (labels == s)].mean(axis=0)
        for s in subjects
    ])

    distances = cdist(centroids, centroids, metric="euclidean")
    np.fill_diagonal(distances, np.inf)

    nn_map = {
        s: subjects[int(np.argmin(distances[i]))]
        for i, s in enumerate(subjects)
    }
    return nn_map, distances


if __name__ == "__main__":
    features, labels, groups = load_data()
    subjects = sorted(np.unique(labels).tolist())

    print(f"Subjects: {len(subjects)}, Features: {features.shape[1]}")

    nn_map, distances = build_nearest_neighbour_map(
        features, labels, groups, subjects)

    with open("summary_exact_eegmmidb.json") as f:
        exact_results = json.load(f)["per_target"]

    print("\n" + "=" * 62)
    print("NEAREST-NEIGHBOUR BASELINE (exact retraining)")
    print("=" * 62)
    print(f"\n{'Classifier':<20}{'Matches':>12}{'Match rate':>14}")
    print("-" * 62)

    summary = {}

    for clf_name in CLF_NAMES:
        results = exact_results.get(clf_name, [])
        if not results:
            print(f"{clf_name:<20}{'N/A':>12}{'N/A':>14}")
            continue

        matches = sum(
            1 for r in results
            if r["top_predicted_class"] == nn_map[r["target_subject"]]
        )
        rate = matches / len(results)
        summary[clf_name] = {
            "matches":    matches,
            "n_subjects": len(results),
            "match_rate": rate,
        }
        print(f"{clf_name:<20}{f'{matches}/{len(results)}':>12}{rate:>13.1%}")

    with open("summary_nn_baseline.json", "w") as f:
        json.dump({
            "nearest_neighbour_map": nn_map,
            "per_classifier":        summary,
        }, f, indent=2)

    print("\nSaved summary_nn_baseline.json")