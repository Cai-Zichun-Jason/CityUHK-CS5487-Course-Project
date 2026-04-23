"""
Compare Linear PCA vs Kernel PCA on classifier accuracy.

Uses sklearn KernelPCA with multiple kernels and hyperparameters.
Runs LR and SVM on Trial 1 across a range of reduced dimensionalities.

Output: results/eval_figures/eval_kernel_pca.pdf
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as SkLR

from utils.data_io import load_trial

N_COMPONENTS_LIST = list(range(10, 161, 10))  # 10, 20, 30, ..., 160

PCA_METHODS = {
    "Linear PCA": lambda n: PCA(n_components=n),
    "KernelPCA (rbf, γ=0.01)": lambda n: KernelPCA(n_components=n, kernel="rbf", gamma=0.01),
    "KernelPCA (rbf, γ=0.001)": lambda n: KernelPCA(n_components=n, kernel="rbf", gamma=0.001),
    "KernelPCA (rbf, γ=0.0001)": lambda n: KernelPCA(n_components=n, kernel="rbf", gamma=0.0001),
    "KernelPCA (poly d=2)": lambda n: KernelPCA(n_components=n, kernel="poly", degree=2),
    "KernelPCA (poly d=3)": lambda n: KernelPCA(n_components=n, kernel="poly", degree=3),
    "KernelPCA (cosine)": lambda n: KernelPCA(n_components=n, kernel="cosine"),
    "KernelPCA (sigmoid)": lambda n: KernelPCA(n_components=n, kernel="sigmoid", gamma=0.001),
}

CLASSIFIERS = {
    "Logistic Regression": lambda: SkLR(max_iter=2000, C=1.0, solver="lbfgs"),
    "SVM (RBF)": lambda: SVC(kernel="rbf", C=10.0, gamma="scale"),
}

MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]
COLORS = [
    "#1f77b4", "#ff7f0e", "#d62728", "#e377c2",
    "#2ca02c", "#8c564b", "#17becf", "#7f7f7f",
]


def main():
    print("Loading Trial 1 ...")
    Xtr, ytr, Xte, yte = load_trial(1)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # results[clf_name][pca_name] = list of acc per n_components
    results = {cn: {pn: [] for pn in PCA_METHODS} for cn in CLASSIFIERS}

    for n_comp in N_COMPONENTS_LIST:
        for pca_name, pca_fn in PCA_METHODS.items():
            pca = pca_fn(n_comp)
            try:
                Xtr_p = pca.fit_transform(Xtr_s)
                Xte_p = pca.transform(Xte_s)
            except Exception as e:
                print(f"  [WARN] {pca_name} n={n_comp}: {e}")
                for cn in CLASSIFIERS:
                    results[cn][pca_name].append(np.nan)
                continue
            for clf_name, clf_fn in CLASSIFIERS.items():
                clf = clf_fn()
                clf.fit(Xtr_p, ytr)
                acc = clf.score(Xte_p, yte)
                results[clf_name][pca_name].append(acc)
                print(f"  dim={n_comp:>3d}  {pca_name:<28s}  {clf_name:<22s}  acc={acc:.4f}")

    # --- Plot ---
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "eval_figures")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(CLASSIFIERS), figsize=(14, 6), sharey=True)
    for ax, (clf_name, pca_dict) in zip(axes, results.items()):
        for idx, (pca_name, accs) in enumerate(pca_dict.items()):
            ax.plot(N_COMPONENTS_LIST, accs,
                    marker=MARKERS[idx % len(MARKERS)],
                    color=COLORS[idx % len(COLORS)],
                    label=pca_name, markersize=4, linewidth=1.2)
        ax.set_title(clf_name, fontsize=12)
        ax.set_xlabel("Reduced Dimensionality (# PCA components)")
        ax.set_ylabel("Test Accuracy")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Linear PCA vs Kernel PCA — Classifier Accuracy",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "eval_kernel_pca.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {out_path}")


if __name__ == "__main__":
    main()
