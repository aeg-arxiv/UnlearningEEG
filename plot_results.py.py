"""
Figure 1: aggregate prediction entropy and MIA AUC across methods and
classifier families on EEGMMIDB.

Values are read from the per-subject summary files, so the figure reflects
whatever results are present rather than hardcoded numbers.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_FILES = {
    "Naive\nDeletion":   "summary_naive_eegmmidb.json",
    "TGP":               "summary_tgp_eegmmidb.json",
    "RLP":               "summary_rlp_eegmmidb.json",
    "SISA":              "summary_sisa_eegmmidb.json",
    "Exact\nRetraining": "summary_exact_eegmmidb.json",
}

CLF_KEYS   = ["RandomForest", "ExtraTrees", "SGDClassifier", "RidgeClassifier"]
CLF_LABELS = ["Random Forest", "Extra Trees", "SGD", "Ridge"]
COLORS     = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]


def load_means():
    """Return (entropy, mia) arrays of shape (n_methods, n_classifiers)."""
    entropy_rows, mia_rows = [], []
    for fname in SUMMARY_FILES.values():
        with open(fname) as f:
            data = json.load(f)
        e_row, m_row = [], []
        for clf in CLF_KEYS:
            res = data["per_target"][clf]
            e_row.append(np.mean(
                [r["prediction_entropy_normalized"] for r in res]))
            m_row.append(np.mean([r["mia_auc"] for r in res]))
        entropy_rows.append(e_row)
        mia_rows.append(m_row)
    return np.array(entropy_rows), np.array(mia_rows)


def style_panel(ax, title, ylabel, methods, ticks, ticklabels, ref_line):
    ax.set_xlabel("Method", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=ref_line, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


if __name__ == "__main__":
    methods = list(SUMMARY_FILES.keys())
    entropy, mia = load_means()

    x       = np.arange(len(methods))
    width   = 0.16
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for i, (color, offset) in enumerate(zip(COLORS, offsets)):
        ax1.bar(x + offset * width, entropy[:, i],
                width, color=color, alpha=0.85)
        ax2.bar(x + offset * width, mia[:, i],
                width, color=color, alpha=0.85)

    style_panel(
        ax1, "(a) Aggregate Prediction Entropy", "Normalized Entropy",
        methods,
        [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0 (ideal)"],
        ref_line=1.0)

    style_panel(
        ax2, "(b) MIA AUC", "MIA AUC",
        methods,
        [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
        ["0.0", "0.2", "0.4", "0.5 (ideal)", "0.6", "0.8", "1.0"],
        ref_line=0.5)

    ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in COLORS]
    fig.legend(handles=handles, labels=CLF_LABELS,
               loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("fig1_results.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("fig1_results.png", bbox_inches="tight", dpi=300)
