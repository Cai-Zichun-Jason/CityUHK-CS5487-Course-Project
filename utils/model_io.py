"""
Save / load trained models (preprocessor + classifier) under data/trained-model/.
CNN uses torch.save (state_dict); other models use joblib.
"""
import os
import pickle
import joblib

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "trained-model",
)


def _path(name, trial, ext):
    return os.path.join(MODEL_DIR, f"{name}_trial{trial}.{ext}")


def save_model(name, trial, clf, pre):
    """Save (preprocessor, classifier) to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if name == "CNN":
        # CNN: save torch state_dict + preprocessor (pickled)
        import torch
        path = _path(name, trial, "pt")
        torch.save({
            "state_dict": clf.model.state_dict(),
            "epochs": clf.epochs,
            "batch_size": clf.batch_size,
            "lr": clf.lr,
            "seed": clf.seed,
            "preprocessor": pickle.dumps(pre),
        }, path)
    else:
        # everything else: joblib
        path = _path(name, trial, "pkl")
        joblib.dump({"clf": clf, "pre": pre}, path)
    return path


def load_model(name, trial):
    """Load (clf, preprocessor). Raises FileNotFoundError if missing."""
    if name == "CNN":
        import torch
        from pipeline.CNN import CNN, build_lenet
        path = _path(name, trial, "pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        st = torch.load(path, map_location="cpu", weights_only=False)
        clf = CNN(epochs=st["epochs"], batch_size=st["batch_size"],
                  lr=st["lr"], seed=st["seed"])
        clf.model = build_lenet().to(clf.device)
        clf.model.load_state_dict(st["state_dict"])
        return clf, pickle.loads(st["preprocessor"])

    path = _path(name, trial, "pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = joblib.load(path)
    return d["clf"], d["pre"]
