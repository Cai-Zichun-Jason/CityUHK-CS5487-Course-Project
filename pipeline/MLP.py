"""MLP classifier (sklearn), 3-hidden-layer dense net.

Architecture and hyper-parameters chosen to match the report:
    hidden = (512, 256, 128), lr = 0.005, batch = 64, max_iter = 120.
"""
from sklearn.neural_network import MLPClassifier


def build():
    return MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=0.005,
        batch_size=64,
        max_iter=120,
        alpha=5e-4,
        shuffle=True,
        random_state=42,
        verbose=False,
    )
