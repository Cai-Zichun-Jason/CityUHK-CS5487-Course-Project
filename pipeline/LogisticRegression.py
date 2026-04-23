"""
Multinomial Logistic Regression with L2 regularization — from scratch
Uses L-BFGS-B optimizer from scipy
"""
import numpy as np
from scipy.optimize import minimize


class SoftmaxLR:
    def __init__(self, C=1.0, max_iter=200, tol=1e-6, seed=42):
        self.C = float(C)
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.W = None
        self.b = None
        self.classes_ = None

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        ez = np.exp(Z)
        return ez / ez.sum(axis=1, keepdims=True)

    def _pack(self, W, b):
        return np.concatenate([W.ravel(), b.ravel()])

    def _unpack(self, theta, K, D):
        W = theta[: K * D].reshape(K, D)
        b = theta[K * D:]
        return W, b

    def _loss_grad(self, theta, X, Y, K, D):
        W, b = self._unpack(theta, K, D)
        N = X.shape[0]
        Z = X @ W.T + b
        P = self._softmax(Z)

        # negative log-likelihood + L2 regularization
        nll = -np.mean(np.log(P[np.arange(N), Y] + 1e-12))
        reg = 0.5 / (self.C * N) * (W * W).sum()
        loss = nll + reg

        # gradient
        dZ = P.copy()
        dZ[np.arange(N), Y] -= 1.0
        dZ /= N
        gW = dZ.T @ X + W / (self.C * N)
        gb = dZ.sum(axis=0)
        return loss, self._pack(gW, gb)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        D = X.shape[1]

        # map labels to 0..K-1
        idx = {c: i for i, c in enumerate(self.classes_)}
        Y = np.array([idx[c] for c in y], dtype=int)

        theta0 = 0.01 * rng.standard_normal(K * D + K)
        result = minimize(
            self._loss_grad, theta0, args=(X, Y, K, D),
            method="L-BFGS-B", jac=True,
            options=dict(maxiter=self.max_iter, gtol=self.tol),
        )
        self.W, self.b = self._unpack(result.x, K, D)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = X @ self.W.T + self.b
        return self.classes_[Z.argmax(axis=1)]


def build():
    return SoftmaxLR(C=1.0, max_iter=300)
