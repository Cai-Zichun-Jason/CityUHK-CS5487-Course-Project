"""
Data I/O for digits4000 (base) and challenge subset.
"""
import os
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"
)


def load_trial(trial_id):
    """
    Load training and test data for trial 1 or 2.
    Input:
        trial_id: 1 or 2
    Return:
        X_train, y_train, X_test, y_test
    """
    assert trial_id in (1, 2), "trial_id must be 1 or 2"
    base = os.path.join(DATA_DIR, "base")

    X = np.loadtxt(os.path.join(base, "digits4000_digits_vec.txt"), dtype=np.float64)
    y = np.loadtxt(os.path.join(base, "digits4000_digits_labels.txt"), dtype=int)
    tr_idx = np.loadtxt(os.path.join(base, "digits4000_trainset.txt"), dtype=int)
    te_idx = np.loadtxt(os.path.join(base, "digits4000_testset.txt"), dtype=int)

    # column 0 -> trial 1, column 1 -> trial 2; convert 1-based to 0-based
    col = trial_id - 1
    tr = tr_idx[:, col] - 1
    te = te_idx[:, col] - 1
    return X[tr], y[tr], X[te], y[te]


def load_challenge():
    """Load the 150-sample challenge test set. Return X, y."""
    ch = os.path.join(DATA_DIR, "challenge")
    X = np.loadtxt(os.path.join(ch, "cdigits_digits_vec.txt"), dtype=np.float64)
    y = np.loadtxt(os.path.join(ch, "cdigits_digits_labels.txt"), dtype=int)
    return X, y
