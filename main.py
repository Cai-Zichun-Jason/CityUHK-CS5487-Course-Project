"""
CS5487 demo entry point.

Modes:
    --mode train      train and save model on both trials
    --mode base       load saved model (train if missing), evaluate on base test set
    --mode challenge  load saved model (train if missing), evaluate on challenge set
    --mode eval       run hyperparameter sensitivity analysis (evaluation/ scripts)

Examples:
    python3 main.py --mode train --model all
    python3 main.py --mode base  --model SVM
    python3 main.py --mode challenge --model CNN
    python3 main.py --mode eval
"""
import argparse
import os
import subprocess
import time

import numpy as np

from pipeline.preprocess import Preprocessor
from pipeline import (OneNN, LogisticRegression, KernelSVM,
                      RandomForest, XGBoost, MLP, CNN)
from utils.data_io import load_trial, load_challenge
from utils.model_io import save_model, load_model
from utils.results_io import (write_summary, save_confusion_matrix,
                              compute_macro_f1, compute_macro_recall)


# model name -> (module, preprocessing mode, pca_variance_ratio)
MODELS = {
    "1NN": (OneNN, "noprocess", 0.95),
    "LR":  (LogisticRegression, "wholeprocess", 0.99),
    "SVM": (KernelSVM, "wholeprocess", 0.95),
    "RF":  (RandomForest, "noPCA", 0.95),
    "XGB": (XGBoost, "noPCA", 0.95),
    "MLP": (MLP, "wholeprocess", 0.95),
    "CNN": (CNN, "noPCA", 0.95),
}

# rough total fit time (seconds, both trials combined), just for the budget banner
TRAIN_TIME_BUDGET_S = {
    "1NN":   1,
    "LR":    1,
    "SVM":  32,
    "RF":    2,
    "XGB": 145,
    "MLP":  10,
    "CNN":  20,
}


def fmt_secs(s):
    """Format seconds as '4.2s' or '1m 23s'."""
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    return f"{m}m {sec:02d}s"


def train_one(name, trial):
    """Train + save one model on one trial. Return (test_acc, fit_time, f1, recall)."""
    mod, pp_mode, pca_var = MODELS[name]
    Xtr, ytr, Xte, yte = load_trial(trial)

    pre = Preprocessor(pp_mode, pca_var=pca_var)
    Xtr_p = pre.fit_transform(Xtr)
    Xte_p = pre.transform(Xte)

    clf = mod.build()
    t0 = time.perf_counter()
    clf.fit(Xtr_p, ytr)
    fit_time = time.perf_counter() - t0

    y_pred = clf.predict(Xte_p)
    acc = float(np.mean(y_pred == yte))
    f1 = compute_macro_f1(yte, y_pred)
    recall = compute_macro_recall(yte, y_pred)
    path = save_model(name, trial, clf, pre)
    print(f"  [train] {name} trial{trial}: acc={acc:.4f}  f1={f1:.4f}  recall={recall:.4f}  "
          f"fit={fmt_secs(fit_time)}  -> {path}")
    return acc, fit_time, f1, recall


def eval_one(name, trial, challenge):
    """Load (or train) model, evaluate on base or challenge set, save confusion matrix."""
    try:
        clf, pre = load_model(name, trial)
    except FileNotFoundError:
        print(f"  [{name} trial{trial}] no saved model -> training first")
        train_one(name, trial)
        clf, pre = load_model(name, trial)

    if challenge:
        Xte, yte = load_challenge()
    else:
        _, _, Xte, yte = load_trial(trial)

    y_pred = clf.predict(pre.transform(Xte))
    acc = float(np.mean(y_pred == yte))
    f1 = compute_macro_f1(yte, y_pred)
    recall = compute_macro_recall(yte, y_pred)

    mode = "challenge" if challenge else "base"
    img = save_confusion_matrix(name, trial, mode, yte, y_pred)
    print(f"  [{mode}] {name} trial{trial}: acc={acc:.4f}  f1={f1:.4f}  recall={recall:.4f}  cm -> {img}")
    return acc, f1, recall


EVAL_SCRIPTS = [
    "evaluation/eval_svm_kernel.py",
    "evaluation/eval_svm_grid.py",
    "evaluation/eval_pca_variance.py",
    "evaluation/eval_kernel_pca.py",
    "evaluation/eval_speed_comparison.py",
    "evaluation/eval_error_analysis.py",
]


def run_eval_scripts():
    """Run all hyperparameter sensitivity analysis scripts in evaluation/."""
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Running {len(EVAL_SCRIPTS)} evaluation scripts ...\n")
    for script in EVAL_SCRIPTS:
        path = os.path.join(demo_dir, script)
        print(f"{'='*56}\n  Running: {script}\n{'='*56}")
        t0 = time.perf_counter()
        subprocess.run(["python3", path], cwd=demo_dir, check=True)
        elapsed = time.perf_counter() - t0
        print(f"  Done in {fmt_secs(elapsed)}\n")
    print(f"All evaluation figures saved to results/eval_figures/")


def run(mode, names):
    print(f"{'='*56}\n  Mode: {mode}   |   Models: {', '.join(names)}\n{'='*56}")

    # show estimated time budget for training
    if mode == "train":
        budget = sum(TRAIN_TIME_BUDGET_S.get(n, 0) for n in names)
        per_model = ", ".join(
            f"{n}~{fmt_secs(TRAIN_TIME_BUDGET_S.get(n, 0))}" for n in names
        )
        print(f"  Estimated total fit time: ~{fmt_secs(budget)}  "
              f"(per model: {per_model})\n")

    wall_start = time.perf_counter()
    summary = {}

    for name in names:
        print(f"\n>>> {name}")
        scores, train_times, f1_scores, recall_scores = [], [], [], []

        for trial in (1, 2):
            if mode == "train":
                acc, fit_time, f1, recall = train_one(name, trial)
                scores.append(acc)
                train_times.append(fit_time)
                f1_scores.append(f1)
                recall_scores.append(recall)
            else:
                acc, f1, recall = eval_one(name, trial, challenge=(mode == "challenge"))
                scores.append(acc)
                f1_scores.append(f1)
                recall_scores.append(recall)

        m, s = float(np.mean(scores)), float(np.std(scores))
        mf1 = float(np.mean(f1_scores))
        mrec = float(np.mean(recall_scores))
        summary[name] = (scores, m, s, train_times, f1_scores, recall_scores)
        msg = f"  -> mean_acc={m:.4f}  std={s:.4f}  F1={mf1:.4f}  Recall={mrec:.4f}"
        if train_times:
            msg += f"  total_fit={fmt_secs(sum(train_times))}"
        print(msg)

        # write to JSON ledger
        if mode == "train":
            path = write_summary(mode, name, scores, train_times, f1_scores, recall_scores)
        else:
            path = write_summary(mode, name, scores,
                                 trial_f1s=f1_scores, trial_recalls=recall_scores)
        print(f"  -> ledger updated: {path}")

    wall_total = time.perf_counter() - wall_start

    # print summary table
    print(f"\n{'='*70}\n  Summary ({mode})\n{'='*70}")
    if mode == "train":
        hdr = f"  {'Model':<6} {'Trial1':>8} {'Trial2':>8} {'Mean':>8} {'Std':>8} {'F1':>8} {'Recall':>8} {'Fit':>8}"
    else:
        hdr = f"  {'Model':<6} {'Trial1':>8} {'Trial2':>8} {'Mean':>8} {'Std':>8} {'F1':>8} {'Recall':>8}"
    print(hdr)
    for name, (scs, m, s, tts, f1s, recs) in summary.items():
        t1 = f"{scs[0]:.4f}" if len(scs) > 0 else ""
        t2 = f"{scs[1]:.4f}" if len(scs) > 1 else ""
        mf1 = f"{np.mean(f1s):.4f}" if f1s else ""
        mrec = f"{np.mean(recs):.4f}" if recs else ""
        row = f"  {name:<6} {t1:>8} {t2:>8} {m:>8.4f} {s:>8.4f} {mf1:>8} {mrec:>8}"
        if mode == "train":
            row += f" {fmt_secs(sum(tts)):>8}"
        print(row)
    print(f"\n  Total wall-clock time: {fmt_secs(wall_total)}")


def main():
    parser = argparse.ArgumentParser(description="CS5487 demo: digit classification.")
    parser.add_argument("--mode", required=True,
                        choices=["train", "base", "challenge", "eval"],
                        help="train, evaluate on base test, evaluate on challenge set, "
                             "or run hyperparameter analysis scripts.")
    parser.add_argument("--model", default="all",
                        choices=list(MODELS) + ["all"],
                        help="which model to run (default: all). Ignored for --mode eval.")
    args = parser.parse_args()

    if args.mode == "eval":
        run_eval_scripts()
    else:
        names = list(MODELS) if args.model == "all" else [args.model]
        run(args.mode, names)


if __name__ == "__main__":
    main()
