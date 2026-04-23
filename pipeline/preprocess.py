"""
Preprocessing pipeline — from-scratch NumPy implementation

Three modes:
    wholeprocess : median filter -> gaussian smooth -> centroid center
                   -> flatten -> normalize -> PCA (used by SVM, LR, MLP)
    noPCA        : same as above but skip PCA (used by CNN, XGBoost, RF)
    noprocess    : just X / 255 (used by 1-NN baseline)
"""
import numpy as np

VALID_MODES = ("noprocess", "noPCA", "wholeprocess")


def median_filter_3x3(img):
    """3x3 median filter, reflect padding."""
    pad = np.pad(img, 1, mode="reflect")
    H, W = img.shape
    stack = np.stack(
        [pad[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)], axis=0
    )
    return np.median(stack, axis=0)


def _gaussian_kernel_1d(sigma):
    """1D Gaussian kernel, radius = ceil(3*sigma)."""
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_filter_2d(img, sigma=1.0):
    """Separable 2D Gaussian smoothing."""
    k = _gaussian_kernel_1d(sigma)
    r = len(k) // 2
    pad = np.pad(img, r, mode="reflect")
    H, W = img.shape
    # convolve rows then columns
    tmp = np.zeros((H, W + 2 * r), dtype=np.float64)
    for i, w in enumerate(k):
        tmp += w * pad[i:i + H, :]
    out = np.zeros((H, W), dtype=np.float64)
    for i, w in enumerate(k):
        out += w * tmp[:, i:i + W]
    return out


def shift_image(img, dy, dx):
    """Bilinear interpolation shift, zero-padded."""
    H, W = img.shape
    yy, xx = np.indices((H, W), dtype=np.float64)
    src_y = yy - dy
    src_x = xx - dx
    y0 = np.floor(src_y).astype(int)
    x0 = np.floor(src_x).astype(int)
    y1, x1 = y0 + 1, x0 + 1
    wy, wx = src_y - y0, src_x - x0

    def gather(yi, xi):
        valid = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
        v = np.zeros_like(img, dtype=np.float64)
        v[valid] = img[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)][valid]
        return v

    return ((1 - wy) * (1 - wx) * gather(y0, x0)
            + (1 - wy) * wx * gather(y0, x1)
            + wy * (1 - wx) * gather(y1, x0)
            + wy * wx * gather(y1, x1))


def centroid_center(img):
    """Shift image so intensity centroid lands at (14, 14)."""
    tot = img.sum()
    if tot <= 0:
        return img
    ys, xs = np.indices(img.shape, dtype=np.float64)
    cy = (ys * img).sum() / tot
    cx = (xs * img).sum() / tot
    dy, dx = 14.0 - cy, 14.0 - cx
    if abs(dy) < 0.1 and abs(dx) < 0.1:
        return img
    return shift_image(img, dy, dx)


class PCA:
    """PCA via covariance eigendecomposition, keep variance_ratio of variance."""

    def __init__(self, variance_ratio=0.95):
        self.variance_ratio = variance_ratio
        self.mean_ = None
        self.components_ = None
        self.n_components_ = 0

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / max(N - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # sort descending
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        # pick k components covering target variance
        ratio = eigvals / (eigvals.sum() + 1e-12)
        k = int(np.searchsorted(np.cumsum(ratio), self.variance_ratio) + 1)
        k = max(1, min(k, len(eigvals)))
        self.components_ = eigvecs[:, :k].T
        self.n_components_ = k
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T


class Preprocessor:
    def __init__(self, mode="wholeprocess", pca_var=0.95, seed=42):
        assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}"
        self.mode = mode
        self.pca_var = pca_var
        self.seed = seed
        self._pca = None
        self._min = None
        self._max = None

    def fit_transform(self, X):
        return self._run(X, fit=True)

    def transform(self, X):
        return self._run(X, fit=False)

    def _run(self, X, fit):
        # 1-NN baseline: no preprocessing
        if self.mode == "noprocess":
            return X.astype(np.float64) / 255.0

        # image domain processing
        imgs = np.asarray(X, dtype=np.float64).reshape(-1, 28, 28)
        imgs = np.stack([median_filter_3x3(img) for img in imgs])
        imgs = np.stack([gaussian_filter_2d(img, 1.0) for img in imgs])
        imgs = np.stack([np.clip(centroid_center(img), 0, 255) for img in imgs])
        V = imgs.reshape(imgs.shape[0], 784)

        # min-max normalize to [0, 1]
        if fit:
            self._min = float(V.min())
            self._max = float(V.max())
        V = np.clip((V - self._min) / (self._max - self._min + 1e-8), 0.0, 1.0)

        # PCA (only in wholeprocess mode)
        if self.mode == "wholeprocess":
            if fit:
                self._pca = PCA(variance_ratio=self.pca_var).fit(V)
            V = self._pca.transform(V)

        return V
