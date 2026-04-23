"""
Effect of PCA variance ratio on classifier accuracy.

Run LogisticRegression and KernelSVM under several PCA variance settings
on Trial 1, plot test accuracy vs variance ratio and number of components.

Output: results/eval_figures/eval_pca_variance.pdf
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.preprocess import Preprocessor
from pipeline.LogisticRegression import SoftmaxLR
from pipeline.KernelSVM import KernelSVM
from utils.data_io import load_trial


VARIANCE_RATIOS = [0.70, 0.80, 0.85, 0.90, 0.95, 0.99]


def run_one(model_name, var_ratio, Xtr_raw, ytr, Xte_raw, yte):
    pre = Preprocessor("wholeprocess", pca_var=var_ratio)
    Xtr = pre.fit_transform(Xtr_raw)
    Xte = pre.transform(Xte_raw)
    n_comp = pre._pca.n_components_

    if model_name == "LR":
        clf = SoftmaxLR(C=1.0, max_iter=300)
    elif model_name == "SVM":
        clf = KernelSVM(C=10.0, gamma="scale", max_passes=5)
    else:
        raise ValueError(model_name)

    clf.fit(Xtr, ytr)
    acc = float(np.mean(clf.predict(Xte) == yte))
    return acc, n_comp


def main():
    print("Loading Trial 1 ...")
    Xtr_raw, ytr, Xte_raw, yte = load_trial(1)

    results = {"LR": [], "SVM": []}
    components = []

    for v in VARIANCE_RATIOS:
        print(f"\nPCA variance = {v}")
        for name in ("LR", "SVM"):
            acc, n_comp = run_one(name, v, Xtr_raw, ytr, Xte_raw, yte)
            results[name].append(acc)
            print(f"  {name}: acc={acc:.4f}  (n_components={n_comp})")
            if name == "LR":
                # n_components only depends on variance ratio, record once
                components.append(n_comp)

    # plot: two y-axes (accuracy and #components)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "eval_figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_pca_variance.pdf")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(VARIANCE_RATIOS, results["LR"], "o-", label="LR", color="tab:blue")
    ax1.plot(VARIANCE_RATIOS, results["SVM"], "s-", label="SVM", color="tab:red")
    ax1.set_xlabel("PCA variance ratio")
    ax1.set_ylabel("Test accuracy (Trial 1)")
    ax1.set_ylim(0.85, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    # secondary axis for n_components
    ax2 = ax1.twinx()
    ax2.bar(VARIANCE_RATIOS, components, width=0.015, alpha=0.2,
            color="gray", label="#components")
    ax2.set_ylabel("# PCA components", color="gray")
    ax2.tick_params(axis="y", colors="gray")

    fig.suptitle("PCA Variance Ratio — Accuracy & Components")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {out_path}")


if __name__ == "__main__":
    main()
