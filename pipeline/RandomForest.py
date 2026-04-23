"""Random Forest classifier (sklearn)."""
from sklearn.ensemble import RandomForestClassifier


def build():
    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
