"""
Kernel SVM (RBF) — from-scratch SMO + One-vs-Rest
"""
import numpy as np

from pipeline.kernels import rbf_kernel


class _BinarySVM:
    """Simplified SMO for one binary SVM, labels in {-1, +1}."""

    def __init__(self, C=10.0, max_passes=5, tol=1e-3, seed=0):
        self.C = C
        self.max_passes = max_passes
        self.tol = tol
        self.seed = seed
        self.alpha = None
        self.b = 0.0

    def fit_with_kernel(self, K, y):
        rng = np.random.default_rng(self.seed)
        n = K.shape[0]
        alpha = np.zeros(n)
        b = 0.0
        f = np.full(n, b)   # f(x_i) cache

        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                Ei = f[i] - y[i]
                if (y[i] * Ei < -self.tol and alpha[i] < self.C) or \
                   (y[i] * Ei > self.tol and alpha[i] > 0):
                    j = i
                    while j == i:
                        j = int(rng.integers(0, n))
                    Ej = f[j] - y[j]
                    ai_old, aj_old = alpha[i], alpha[j]

                    if y[i] != y[j]:
                        L = max(0.0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0.0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)
                    if L == H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    aj_new = np.clip(aj_old - y[j] * (Ei - Ej) / eta, L, H)
                    if abs(aj_new - aj_old) < 1e-5:
                        continue
                    ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)
                    alpha[i], alpha[j] = ai_new, aj_new

                    da_i = (ai_new - ai_old) * y[i]
                    da_j = (aj_new - aj_old) * y[j]
                    b1 = b - Ei - da_i * K[i, i] - da_j * K[i, j]
                    b2 = b - Ej - da_i * K[i, j] - da_j * K[j, j]
                    if 0 < ai_new < self.C:
                        b_new = b1
                    elif 0 < aj_new < self.C:
                        b_new = b2
                    else:
                        b_new = 0.5 * (b1 + b2)
                    f += da_i * K[i] + da_j * K[j] + (b_new - b)
                    b = b_new
                    num_changed += 1
            passes = passes + 1 if num_changed == 0 else 0

        self.alpha, self.b = alpha, b


class KernelSVM:
    """Multi-class RBF Kernel SVM (One-vs-Rest)."""

    def __init__(self, C=10.0, gamma="scale", max_passes=5, tol=1e-3, seed=42):
        self.C = C
        self.gamma = gamma
        self.max_passes = max_passes
        self.tol = tol
        self.seed = seed
        self.classes_ = None
        self.X_train = None
        self.alpha_y = None
        self.b = None
        self._gamma = None

    def _resolve_gamma(self, X):
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        if self.gamma == "scale":
            v = X.var()
            return 1.0 / (X.shape[1] * v) if v > 0 else 1.0 / X.shape[1]
        return 1.0 / X.shape[1]  # 'auto'

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)

        self._gamma = self._resolve_gamma(X)
        K = rbf_kernel(X, X, self._gamma)

        n = X.shape[0]
        self.X_train = X
        self.alpha_y = np.zeros((n_cls, n))
        self.b = np.zeros(n_cls)

        for k, c in enumerate(self.classes_):
            yk = np.where(y == c, 1.0, -1.0)
            svm = _BinarySVM(C=self.C, max_passes=self.max_passes,
                             tol=self.tol, seed=self.seed + k)
            svm.fit_with_kernel(K, yk)
            self.alpha_y[k] = svm.alpha * yk
            self.b[k] = svm.b

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        K = rbf_kernel(X, self.X_train, self._gamma)
        return K @ self.alpha_y.T + self.b

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def grid_search_svm(X, y, n_splits=5):
    """
    Grid search for SVM hyperparameters (C, gamma)
    Input:
        X: training features, already preprocessed (N, D)
        y: training labels (N,)
        n_splits: number of cross validation folds
    Return:
        best_C, best_gamma
    """
    # search space
    C_candidates = [0.1, 1.0, 10.0, 100.0]
    gamma_candidates = ["scale", "auto", 0.01, 0.001]

    # k-fold split
    n = len(y)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    folds = np.array_split(indices, n_splits)

    best_score = -1
    best_C = None
    best_gamma = None

    print(f"Grid search: {len(C_candidates) * len(gamma_candidates)} combinations, "
          f"{n_splits}-fold CV")

    for C in C_candidates:
        for gamma in gamma_candidates:
            fold_scores = []
            for i in range(n_splits):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])

                clf = KernelSVM(C=C, gamma=gamma, max_passes=5)
                clf.fit(X[train_idx], y[train_idx])
                pred = clf.predict(X[val_idx])
                acc = np.mean(pred == y[val_idx])
                fold_scores.append(acc)

            mean_score = np.mean(fold_scores)
            print(f"  C={C}, gamma={gamma}: CV acc={mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_gamma = gamma

    print(f"Best: C={best_C}, gamma={best_gamma}, CV acc={best_score:.4f}")
    return best_C, best_gamma


def build(C=10.0, gamma="scale"):
    return KernelSVM(C=C, gamma=gamma, max_passes=5)
