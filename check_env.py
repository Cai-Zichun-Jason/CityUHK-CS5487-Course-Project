#!/usr/bin/env python3
"""
Environment check for the demo: verify all required libraries and data files.
Usage: python3 check_env.py
"""
import os
import sys


def check_pkg(name, import_name=None):
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  [OK]   {name:15s} {ver}")
        return True
    except ImportError:
        print(f"  [FAIL] {name:15s} not installed")
        return False


def main():
    print("=" * 50)
    print("CS5487 Demo — Environment Check")
    print("=" * 50)

    print("\nRequired packages:")
    required = [
        ("numpy",        "numpy"),
        ("scipy",        "scipy"),
        ("scikit-learn", "sklearn"),
        ("xgboost",      "xgboost"),
        ("torch",        "torch"),
        ("joblib",       "joblib"),
        ("matplotlib",   "matplotlib"),
    ]
    pkg_ok = all(check_pkg(name, imp) for name, imp in required)

    print("\nData files:")
    here = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(here, "data", "raw", "base")
    challenge_dir = os.path.join(here, "data", "raw", "challenge")

    base_files = [
        "digits4000_digits_vec.txt",
        "digits4000_digits_labels.txt",
        "digits4000_trainset.txt",
        "digits4000_testset.txt",
    ]
    challenge_files = [
        "cdigits_digits_vec.txt",
        "cdigits_digits_labels.txt",
    ]

    data_ok = True
    for f in base_files:
        path = os.path.join(base_dir, f)
        if os.path.exists(path):
            print(f"  [OK]   data/raw/base/{f}")
        else:
            print(f"  [FAIL] data/raw/base/{f}  -- missing")
            data_ok = False
    for f in challenge_files:
        path = os.path.join(challenge_dir, f)
        if os.path.exists(path):
            print(f"  [OK]   data/raw/challenge/{f}")
        else:
            print(f"  [FAIL] data/raw/challenge/{f}  -- missing")
            data_ok = False

    print("\n" + "=" * 50)
    if pkg_ok and data_ok:
        print("All set. You can run: make train / make base / make eval / make challenge")
        return 0
    if not pkg_ok:
        print("Missing packages -> pip install -r requirements.txt")
    if not data_ok:
        print("Missing data files -> place them under data/raw/")
    return 1


if __name__ == "__main__":
    sys.exit(main())
