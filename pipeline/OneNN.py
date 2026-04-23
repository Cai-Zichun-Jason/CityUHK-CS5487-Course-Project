"""
1-NN classifier — pure NumPy, no sklearn
"""
import numpy as np


class OneNN:
    def __init__(self, k=1):
        self.k = k
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=int)

    def predict(self, X):
        Xt = np.asarray(X, dtype=np.float64)
        sq_train = (self._X ** 2).sum(axis=1)
        out = np.empty(len(Xt), dtype=int)

        # predict in chunks to save memory
        chunk = 512
        for s in range(0, len(Xt), chunk):
            block = Xt[s:s + chunk]
            sq_test = (block ** 2).sum(axis=1)[:, None]
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            d2 = sq_test + sq_train[None, :] - 2.0 * block @ self._X.T
            if self.k == 1:
                out[s:s + chunk] = self._y[d2.argmin(axis=1)]
            else:
                idx = np.argpartition(d2, self.k, axis=1)[:, :self.k]
                neigh = self._y[idx]
                out[s:s + chunk] = np.array(
                    [np.bincount(row, minlength=int(self._y.max()) + 1).argmax()
                     for row in neigh]
                )
        return out


def build():
    return OneNN(k=1)
