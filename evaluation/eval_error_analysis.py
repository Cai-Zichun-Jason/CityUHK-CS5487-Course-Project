"""
Error pattern analysis: per-class F1, recall, and common misclassifications.

Loads saved models, computes predictions on both trials (base + challenge),
then outputs:
  - Macro F1-Score per model (for the report tables)
  - Per-class recall heatmap
  - Top confused digit pairs

Output: results/eval_figures/eval_error_analysis.pdf
        results/error_analysis.json
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.data_io import load_trial, load_challenge
from utils.model_io import load_model

MODEL_NAMES = ["1NN", "LR", "SVM", "RF", "XGB", "MLP", "CNN"]
N_CLASSES = 10


def confusion_matrix(y_true, y_pred, n=N_CLASSES):
    """Build confusion matrix the dumb way."""
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm):
    """Compute per-class precision, recall, f1 from confusion matrix."""
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)

    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        else:
            f1[c] = 0.0

    return precision, recall, f1


def top_confusions(cm, top_k=5):
    """Find top-k off-diagonal (true, pred) pairs by count."""
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], i, j))
    pairs.sort(reverse=True)
    return pairs[:top_k]


def run_one_dataset(dataset_name, load_fn):
    """Run all models on one dataset, return results dict."""
    print(f"\n{'='*50}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*50}")

    results = {}

    for name in MODEL_NAMES:
        trial_f1s = []
        all_confusions = []  # accumulate across trials
        combined_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)

        for trial in (1, 2):
            try:
                clf, pre = load_model(name, trial)
            except FileNotFoundError:
                print(f"  {name} trial{trial}: model not found, skipping")
                continue

            if dataset_name == "challenge":
                Xte, yte = load_challenge()
            else:
                _, _, Xte, yte = load_trial(trial)

            y_pred = clf.predict(pre.transform(Xte))
            cm = confusion_matrix(yte, y_pred)
            combined_cm += cm

            prec, rec, f1 = per_class_metrics(cm)
            macro_f1 = float(f1.mean())
            trial_f1s.append(macro_f1)

            print(f"  {name} trial{trial}: macro_f1={macro_f1:.4f}")

        if not trial_f1s:
            continue

        # combined confusion matrix for error analysis
        prec, rec, f1 = per_class_metrics(combined_cm)
        tops = top_confusions(combined_cm, top_k=5)

        results[name] = {
            "trial_f1": [round(x, 4) for x in trial_f1s],
            "mean_f1": round(float(np.mean(trial_f1s)), 4),
            "per_class_recall": [round(float(x), 4) for x in rec],
            "per_class_f1": [round(float(x), 4) for x in f1],
            "top_confusions": [(int(cnt), int(t), int(p)) for cnt, t, p in tops],
        }

    return results


def plot_recall_heatmap(results, dataset_name, ax):
    """Plot per-class recall heatmap for all models."""
    models = [m for m in MODEL_NAMES if m in results]
    data = np.array([results[m]["per_class_recall"] for m in models])

    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0.6, vmax=1.0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(range(N_CLASSES))
    ax.set_xlabel("Digit")
    ax.set_title(f"Per-class Recall — {dataset_name}")

    for i in range(len(models)):
        for j in range(N_CLASSES):
            val = data[i, j]
            color = "white" if val < 0.75 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)
    return im


def main():
    base_results = run_one_dataset("base", load_trial)
    challenge_results = run_one_dataset("challenge", load_challenge)

    # save JSON
    demo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(demo_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    all_data = {"base": base_results, "challenge": challenge_results}
    json_path = os.path.join(out_dir, "error_analysis.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nJSON saved to: {json_path}")

    # print summary table for LaTeX
    print(f"\n{'='*60}")
    print("  Macro F1-Score (for LaTeX tables)")
    print(f"{'='*60}")
    for ds_name, res in all_data.items():
        print(f"\n  {ds_name}:")
        print(f"  {'Model':<6} {'Trial1':>8} {'Trial2':>8} {'Mean':>8}")
        for name in MODEL_NAMES:
            if name in res:
                r = res[name]
                t1 = f"{r['trial_f1'][0]:.4f}" if len(r['trial_f1']) > 0 else ""
                t2 = f"{r['trial_f1'][1]:.4f}" if len(r['trial_f1']) > 1 else ""
                print(f"  {name:<6} {t1:>8} {t2:>8} {r['mean_f1']:>8.4f}")

    # print top confusions
    print(f"\n{'='*60}")
    print("  Top Confused Digit Pairs (combined trials)")
    print(f"{'='*60}")
    for ds_name, res in all_data.items():
        print(f"\n  {ds_name}:")
        for name in MODEL_NAMES:
            if name in res:
                tops = res[name]["top_confusions"]
                pairs_str = ", ".join(f"{t}→{p} ({c})" for c, t, p in tops[:3])
                print(f"  {name:<6} {pairs_str}")

    # plot
    fig_dir = os.path.join(out_dir, "eval_figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im1 = plot_recall_heatmap(base_results, "Base", axes[0])
    im2 = plot_recall_heatmap(challenge_results, "Challenge", axes[1])
    fig.colorbar(im2, ax=axes, fraction=0.02, pad=0.04)
    fig.suptitle("Per-class Recall by Model", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(fig_dir, "eval_error_analysis.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {out_path}")


if __name__ == "__main__":
    main()
