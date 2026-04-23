"""
Compare Linear SVM vs Kernel SVM (RBF, Poly, Sigmoid) on Trial 1.

Sweeps C values for each kernel and plots accuracy curves.

Output: results/eval_figures/eval_svm_kernel.pdf
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from pipeline.preprocess import Preprocessor
from utils.data_io import load_trial

C_LIST = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
KERNELS = {
    "Linear":       lambda C: SVC(kernel="linear", C=C),
    "RBF":          lambda C: SVC(kernel="rbf", C=C, gamma="scale"),
    "Poly (d=2)":   lambda C: SVC(kernel="poly", C=C, degree=2, gamma="scale"),
    "Poly (d=3)":   lambda C: SVC(kernel="poly", C=C, degree=3, gamma="scale"),
    "Sigmoid":      lambda C: SVC(kernel="sigmoid", C=C, gamma="scale"),
}
MARKERS = ["o", "s", "^", "v", "D"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def main():
    print("Loading and preprocessing Trial 1 ...")
    Xtr, ytr, Xte, yte = load_trial(1)
    pre = Preprocessor("wholeprocess")
    Xtr_p = pre.fit_transform(Xtr)
    Xte_p = pre.transform(Xte)

    results = {k: [] for k in KERNELS}
    for C in C_LIST:
        for kname, kfn in KERNELS.items():
            clf = kfn(C)
            clf.fit(Xtr_p, ytr)
            acc = float(np.mean(clf.predict(Xte_p) == yte))
            results[kname].append(acc)
            print(f"  C={C:<8g}  kernel={kname:<12s}  acc={acc:.4f}")

    # --- Plot ---
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "eval_figures")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (kname, accs) in enumerate(results.items()):
        ax.plot(C_LIST, accs, marker=MARKERS[idx], color=COLORS[idx],
                label=kname, linewidth=1.5, markersize=5)

    ax.set_xscale("log")
    ax.set_xlabel("Regularization parameter C (log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("SVM Kernel Comparison — Accuracy vs C")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "eval_svm_kernel.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {out_path}")


if __name__ == "__main__":
    main()
