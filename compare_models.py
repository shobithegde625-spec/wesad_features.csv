"""Reproducible comparison using shared estimator definitions (historical group5 protocol)."""
from evaluate import main

if __name__ == "__main__":
    main(default_protocol="group5", default_baseline_only=True)
