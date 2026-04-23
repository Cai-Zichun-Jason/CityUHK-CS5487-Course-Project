"""LeNet-style CNN (PyTorch) for 28x28 grayscale digit classification.

The classifier expects 784-d input vectors (no PCA). It internally reshapes
them to 1x28x28 and trains with simple random pixel-shift augmentation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def build_lenet(n_classes=10):
    """Two-stage Conv-BN-ReLU + MaxPool with dropout, then a FC head."""
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, n_classes),
    )


def _augment(batch):
    """Random integer pixel shift in [-2, +2] along both axes (50% rate)."""
    out = torch.zeros_like(batch)
    for i in range(batch.shape[0]):
        dy = int(torch.randint(-2, 3, (1,)).item())
        dx = int(torch.randint(-2, 3, (1,)).item())
        src = batch[i, 0]
        sy1, sy2 = max(0, dy), min(28, 28 + dy)
        sx1, sx2 = max(0, dx), min(28, 28 + dx)
        dy1, dx1 = max(0, -dy), max(0, -dx)
        h, w = sy2 - sy1, sx2 - sx1
        out[i, 0, dy1:dy1 + h, dx1:dx1 + w] = src[sy1:sy2, sx1:sx2]
    return out


class CNN:
    """Minimal sklearn-style wrapper around the LeNet CNN."""

    def __init__(self, epochs=50, batch_size=64, lr=0.001, seed=42):
        self.epochs, self.batch_size, self.lr, self.seed = epochs, batch_size, lr, seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        self.model = build_lenet().to(self.device)
        Xt = torch.FloatTensor(X.reshape(-1, 1, 28, 28))
        yt = torch.LongTensor(y.astype(int))
        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=self.batch_size, shuffle=True)

        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                if torch.rand(1).item() > 0.5:
                    Xb = _augment(Xb)
                opt.zero_grad()
                loss = crit(self.model(Xb), yb)
                loss.backward()
                opt.step()
            sch.step()

    def predict(self, X):
        self.model.eval()
        Xt = torch.FloatTensor(X.reshape(-1, 1, 28, 28))
        preds = []
        with torch.no_grad():
            for s in range(0, len(Xt), self.batch_size):
                logits = self.model(Xt[s:s + self.batch_size].to(self.device))
                preds.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(preds)


def build():
    return CNN()
