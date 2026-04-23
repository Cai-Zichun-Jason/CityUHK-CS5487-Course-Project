"""
Visualize SVM grid search: heatmap of CV accuracy over (C, gamma) on Trial 1.

Uses sklearn SVC for speed; evaluation scripts are allowed to call library APIs.

Output: results/eval_figures/eval_svm_grid.pdf
"""
import os
import sys
import numpy as np

# allow `python evaluation/svm_grid_heatmap.py` from demo/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pipeline.preprocess import Preprocessor
from utils.data_io import load_trial


C_LIST = [0.01, 0.1, 1.0, 10.0, 100.0]
GAMMA_LIST = ["scale", "auto", 0.01, 0.001, 0.0001]
N_SPLITS = 5


def main():
    print("Loading and preprocessing Trial 1 ...")
    Xtr, ytr, _, _ = load_trial(1)
    pre = Preprocessor("wholeprocess")
    Xtr_p = pre.fit_transform(Xtr)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    print(f"Running {len(C_LIST) * len(GAMMA_LIST)} combinations, {N_SPLITS}-fold CV (sklearn SVC)")
    grid = np.zeros((len(C_LIST), len(GAMMA_LIST)))
    for i, C in enumerate(C_LIST):
        for j, gamma in enumerate(GAMMA_LIST):
            clf = SVC(kernel="rbf", C=C, gamma=gamma)
            scores = cross_val_score(clf, Xtr_p, ytr, cv=cv, scoring="accuracy")
            grid[i, j] = scores.mean()
            print(f"  C={C}, gamma={gamma}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # find best
    bi, bj = np.unravel_index(grid.argmax(), grid.shape)
    print(f"\nBest: C={C_LIST[bi]}, gamma={GAMMA_LIST[bj]}, CV acc={grid[bi, bj]:.4f}")

    # plot heatmap
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "eval_figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_svm_grid.pdf")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(GAMMA_LIST)))
    ax.set_yticks(range(len(C_LIST)))
    ax.set_xticklabels([str(g) for g in GAMMA_LIST])
    ax.set_yticklabels([str(c) for c in C_LIST])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title(f"RBF SVM — 5-fold CV Grid Search\n"
                 f"Best: C={C_LIST[bi]}, γ={GAMMA_LIST[bj]}, "
                 f"acc={grid[bi, bj]:.4f}")

    for i in range(len(C_LIST)):
        for j in range(len(GAMMA_LIST)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                    color="black", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to: {out_path}")


if __name__ == "__main__":
    main()
