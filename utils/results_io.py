"""
Persistence for evaluation results: JSON ledger + confusion matrix images.

Files written under demo/results/:
    evaluation.json            - results from --mode eval / train
    evaluation-challenge.json  - results from --mode challenge
    images/                    - confusion matrix per (model, trial)
"""
import json
import os
from datetime import datetime

import numpy as np

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")


def compute_macro_f1(y_true, y_pred, n_classes=10):
    """Compute macro-averaged F1-Score from raw predictions."""
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def compute_macro_recall(y_true, y_pred, n_classes=10):
    """Compute macro-averaged Recall from raw predictions."""
    recalls = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return float(np.mean(recalls))


def _ledger_path(mode):
    # train/base share evaluation.json; challenge has its own
    name = "evaluation-challenge.json" if mode == "challenge" else "evaluation.json"
    return os.path.join(RESULTS_DIR, name)


def _load(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def write_summary(mode, model_name, trial_accs, trial_train_times=None,
                  trial_f1s=None, trial_recalls=None):
    """
    Save (mean, std, per-trial) accuracies, F1-Score, and Recall for one model.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _ledger_path(mode)
    ledger = _load(path)

    entry = ledger.get(model_name, {})
    for i, acc in enumerate(trial_accs, start=1):
        entry[f"trial_{i}_acc"] = round(float(acc), 4)
    entry["mean_acc"] = round(float(np.mean(trial_accs)), 4)
    entry["std_acc"] = round(float(np.std(trial_accs)), 4)

    if trial_f1s is not None:
        for i, f1 in enumerate(trial_f1s, start=1):
            entry[f"trial_{i}_f1"] = round(float(f1), 4)
        entry["mean_f1"] = round(float(np.mean(trial_f1s)), 4)

    if trial_recalls is not None:
        for i, rec in enumerate(trial_recalls, start=1):
            entry[f"trial_{i}_recall"] = round(float(rec), 4)
        entry["mean_recall"] = round(float(np.mean(trial_recalls)), 4)

    if trial_train_times is not None:
        for i, t in enumerate(trial_train_times, start=1):
            entry[f"trial_{i}_train_time"] = round(float(t), 2)

    # only eval/challenge need timestamp; train can be reproduced from saved model
    if mode != "train":
        entry["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elif "last_updated" in entry:
        del entry["last_updated"]
    ledger[model_name] = entry

    with open(path, "w") as f:
        json.dump(ledger, f, indent=2)
    return path


def save_confusion_matrix(model_name, trial, mode, y_true, y_pred, n_classes=10):
    """Save confusion matrix heatmap for one (model, trial, mode)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    os.makedirs(IMAGES_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    suffix = "_challenge" if mode == "challenge" else ""
    ax.set_title(f"{model_name} trial{trial}{suffix}")

    # annotate cells (white text on dark background, black on light)
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, f"{model_name}_trial{trial}{suffix}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
