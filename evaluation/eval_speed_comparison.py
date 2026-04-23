"""
From-scratch vs library speed & accuracy comparison.

Benchmarks on Trial 1 with identical preprocessing:
  - Logistic Regression: from-scratch (SoftmaxLR) vs sklearn
  - Kernel SVM:          from-scratch (SMO OvR)   vs sklearn SVC
  - PCA:                 from-scratch              vs sklearn PCA

Output: results/eval_figures/eval_speed_comparison.pdf
        results/speed_comparison.json
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.data_io import load_trial


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _preprocess_no_pca(X, fit=True, state={}):
    """Image-domain preprocessing + normalize, no PCA. Returns flat (N, 784)."""
    from pipeline.preprocess import (
        median_filter_3x3, gaussian_filter_2d, centroid_center,
    )
    imgs = np.asarray(X, dtype=np.float64).reshape(-1, 28, 28)
    imgs = np.stack([median_filter_3x3(img) for img in imgs])
    imgs = np.stack([gaussian_filter_2d(img, 1.0) for img in imgs])
    imgs = np.stack([np.clip(centroid_center(img), 0, 255) for img in imgs])
    V = imgs.reshape(imgs.shape[0], 784)
    if fit:
        state["min"] = float(V.min())
        state["max"] = float(V.max())
    V = np.clip((V - state["min"]) / (state["max"] - state["min"] + 1e-8), 0.0, 1.0)
    return V


def bench(label, fn):
    """Run fn(), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    print(f"  {label:<35s} {elapsed:>8.3f}s")
    return result, elapsed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("Loading Trial 1 ...")
    Xtr_raw, ytr, Xte_raw, yte = load_trial(1)

    # shared preprocessing (no PCA)
    state = {}
    Xtr_flat = _preprocess_no_pca(Xtr_raw, fit=True, state=state)
    Xte_flat = _preprocess_no_pca(Xte_raw, fit=False, state=state)

    results = {}

    # ===== PCA =====
    print("\n[PCA — 0.95 variance]")
    from pipeline.preprocess import PCA as ScratchPCA
    from sklearn.decomposition import PCA as SkPCA

    def _scratch_pca():
        pca = ScratchPCA(variance_ratio=0.95)
        pca.fit(Xtr_flat)
        Xtr_p = pca.transform(Xtr_flat)
        Xte_p = pca.transform(Xte_flat)
        return Xtr_p, Xte_p, pca.n_components_

    def _sklearn_pca():
        pca = SkPCA(n_components=0.95, random_state=42)
        pca.fit(Xtr_flat)
        Xtr_p = pca.transform(Xtr_flat)
        Xte_p = pca.transform(Xte_flat)
        return Xtr_p, Xte_p, pca.n_components_

    (Xtr_s, Xte_s, nc_s), t_pca_scratch = bench("From-scratch (eigh)", _scratch_pca)
    (Xtr_l, Xte_l, nc_l), t_pca_sklearn = bench("sklearn PCA", _sklearn_pca)
    print(f"  Components: scratch={nc_s}, sklearn={nc_l}")
    results["PCA"] = {
        "From-scratch": {"time": round(t_pca_scratch, 4), "components": int(nc_s)},
        "sklearn":      {"time": round(t_pca_sklearn, 4), "components": int(nc_l)},
        "speedup":      round(t_pca_scratch / max(t_pca_sklearn, 1e-6), 1),
    }

    # Use scratch PCA output for classifiers (consistent with demo pipeline)
    Xtr_pca, Xte_pca = Xtr_s, Xte_s

    # ===== Logistic Regression =====
    print("\n[Logistic Regression — C=1.0]")
    from pipeline.LogisticRegression import SoftmaxLR
    from sklearn.linear_model import LogisticRegression as SkLR

    def _scratch_lr():
        clf = SoftmaxLR(C=1.0, max_iter=300)
        clf.fit(Xtr_pca, ytr)
        return float(np.mean(clf.predict(Xte_pca) == yte))

    def _sklearn_lr():
        clf = SkLR(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr_pca, ytr)
        return float(np.mean(clf.predict(Xte_pca) == yte))

    acc_s, t_s = bench("From-scratch (L-BFGS-B)", _scratch_lr)
    acc_l, t_l = bench("sklearn LogisticRegression", _sklearn_lr)
    print(f"  Accuracy: scratch={acc_s:.4f}, sklearn={acc_l:.4f}")
    results["Logistic Regression"] = {
        "From-scratch": {"acc": acc_s, "time": round(t_s, 3)},
        "sklearn":      {"acc": acc_l, "time": round(t_l, 3)},
        "speedup":      round(t_s / max(t_l, 1e-6), 1),
    }

    # ===== Kernel SVM =====
    print("\n[Kernel SVM — RBF, C=10, gamma=scale]")
    from pipeline.KernelSVM import KernelSVM
    from sklearn.svm import SVC

    def _scratch_svm():
        clf = KernelSVM(C=10.0, gamma="scale", max_passes=5)
        clf.fit(Xtr_pca, ytr)
        return float(np.mean(clf.predict(Xte_pca) == yte))

    def _sklearn_svm():
        clf = SVC(kernel="rbf", C=10.0, gamma="scale")
        clf.fit(Xtr_pca, ytr)
        return float(np.mean(clf.predict(Xte_pca) == yte))

    acc_s, t_s = bench("From-scratch (SMO, OvR)", _scratch_svm)
    acc_l, t_l = bench("sklearn SVC", _sklearn_svm)
    print(f"  Accuracy: scratch={acc_s:.4f}, sklearn={acc_l:.4f}")
    results["Kernel SVM"] = {
        "From-scratch": {"acc": acc_s, "time": round(t_s, 3)},
        "sklearn":      {"acc": acc_l, "time": round(t_l, 3)},
        "speedup":      round(t_s / max(t_l, 1e-6), 1),
    }

    # ===== Random Forest (from project logs) =====
    print("\n[Random Forest — from project logs]")
    demo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    legacy_path = os.path.join(demo_dir, "results", "implementation_vs_library.json")
    if os.path.exists(legacy_path):
        with open(legacy_path) as f:
            legacy = json.load(f)
        if "Random Forest" in legacy:
            rf = legacy["Random Forest"]
            t_rf_s = rf["Our RF (from scratch)"]["mean_train_time"]
            t_rf_l = rf["sklearn RF"]["mean_train_time"]
            acc_rf_s = rf["Our RF (from scratch)"]["mean_acc"]
            acc_rf_l = rf["sklearn RF"]["mean_acc"]
            results["Random Forest"] = {
                "From-scratch": {"acc": acc_rf_s, "time": round(t_rf_s, 3)},
                "sklearn":      {"acc": acc_rf_l, "time": round(t_rf_l, 3)},
                "speedup":      round(t_rf_s / max(t_rf_l, 1e-6), 1),
            }
            print(f"  From-scratch RF              {t_rf_s:>8.3f}s  acc={acc_rf_s:.4f}")
            print(f"  sklearn RF                   {t_rf_l:>8.3f}s  acc={acc_rf_l:.4f}")
    else:
        print("  (implementation_vs_library.json not found, skipping RF)")

    # ===== Save JSON =====
    demo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(demo_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "speed_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to: {json_path}")

    # ===== Print summary =====
    print(f"\n{'='*60}")
    print(f"  {'Component':<22s} {'Scratch':>10s} {'Library':>10s} {'Slowdown':>10s}")
    print(f"{'='*60}")
    for name, r in results.items():
        ts = r["From-scratch"]["time"]
        tl = r["sklearn"]["time"]
        sp = r["speedup"]
        print(f"  {name:<22s} {ts:>9.3f}s {tl:>9.3f}s {sp:>8.1f}×")

    # ===== Plot =====
    fig_dir = os.path.join(out_dir, "eval_figures")
    os.makedirs(fig_dir, exist_ok=True)

    models = list(results.keys())
    scratch_times = [results[m]["From-scratch"]["time"] for m in models]
    lib_times = [results[m]["sklearn"]["time"] for m in models]

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, scratch_times, width,
                   label="From-scratch", color="#d62728")
    bars2 = ax.bar(x + width / 2, lib_times, width,
                   label="Sklearn", color="#1f77b4")

    ax.set_ylabel("Training Time (seconds, log scale)")
    ax.set_title("From-Scratch vs Sklearn — Training Speed")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # annotate speedup
    for i, m in enumerate(models):
        sp = results[m]["speedup"]
        y_pos = max(scratch_times[i], lib_times[i]) * 1.5
        ax.text(i, y_pos, f"{sp}×",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "eval_speed_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {out_path}")


if __name__ == "__main__":
    main()
