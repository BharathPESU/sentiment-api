# src/monitoring/drift_check.py
"""Compare current review lengths with the training data.

Usage:
    python src/monitoring/drift_check.py data/current.csv
"""
import sys

import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_PATH = "data/train.csv"
ALPHA = 0.05  # significance level

def word_counts(df: pd.DataFrame) -> pd.Series:
    return df["text"].astype(str).str.split().str.len()

def main() -> int:
    current_path = sys.argv[1] if len(sys.argv) > 1 else "data/current.csv"
    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(current_path)

    stat, p_value = ks_2samp(word_counts(reference), word_counts(current))
    print(f"Review length KS statistic={stat:.4f}, p-value={p_value:.4g}")

    if p_value < ALPHA:
        print("DRIFT DETECTED: review length distribution changed.")
        return 1
    print("No significant drift in review length.")
    return 0

if __name__ == "__main__":
    sys.exit(main())