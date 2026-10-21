"""Week 01 - Student Data Processing with Pandas and NumPy.

Loads a raw student dataset, cleans it, computes summary statistics, classifies
each student as Pass/Fail and writes the cleaned dataset back to disk.

Run it with:
    python student_data_processing.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE_CSV = DATA_DIR / "students.csv"
OUTPUT_CSV = DATA_DIR / "cleaned_students.csv"
PASS_MARK = 50


def load_data(path: Path = SOURCE_CSV) -> pd.DataFrame:
    """Read the raw student CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.name}")
    return df


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicated student records."""
    cleaned = data.drop_duplicates()
    print(f"Removed {len(data) - len(cleaned)} duplicate row(s)")
    return cleaned


def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing age with the median age and a missing score with the mean."""
    cleaned = data.copy()
    median_age = cleaned["age"].median()
    mean_score = cleaned["score"].mean()
    cleaned["age"] = cleaned["age"].fillna(median_age)
    cleaned["score"] = cleaned["score"].fillna(mean_score)
    print(f"Filled missing age with median ({median_age}) and score with mean ({round(mean_score, 2)})")
    return cleaned


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline: de-duplicate, then impute missing values."""
    return fill_missing_values(remove_duplicates(data))


def classify_students(data: pd.DataFrame, pass_mark: int = PASS_MARK) -> pd.DataFrame:
    """Add a `result` column marking each student as Pass or Fail."""
    classified = data.copy()
    classified["result"] = np.where(classified["score"] >= pass_mark, "Pass", "Fail")
    return classified


def compute_statistics(data: pd.DataFrame) -> dict:
    """Compute summary statistics for the score column using NumPy."""
    scores = data["score"].to_numpy()
    return {
        "average_score": float(np.mean(scores)),
        "highest_score": float(np.max(scores)),
        "lowest_score": float(np.min(scores)),
        "standard_deviation": float(np.std(scores)),
    }


def save_data(data: pd.DataFrame, path: Path = OUTPUT_CSV) -> None:
    """Write the processed dataset to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    print(f"Cleaned dataset saved to {path.name}")


def main() -> None:
    """Run the full pipeline: load, clean, analyse, classify and save."""
    parser = argparse.ArgumentParser(description="Process the student dataset.")
    parser.add_argument("--input", type=Path, default=SOURCE_CSV, help="Path to the raw CSV file.")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Path for the cleaned CSV file.")
    args = parser.parse_args()

    df = load_data(args.input)
    print("\nRaw data:")
    print(df)

    cleaned_df = clean_data(df)
    print("\nCleaned data:")
    print(cleaned_df)
    print("\nRemaining missing values:")
    print(cleaned_df.isnull().sum())

    stats = compute_statistics(cleaned_df)
    print("\nScore statistics:")
    print(f"Average Score: {round(stats['average_score'], 2)}")
    print(f"Highest Score: {stats['highest_score']}")
    print(f"Lowest Score: {stats['lowest_score']}")
    print(f"Standard Deviation: {round(stats['standard_deviation'], 2)}")

    final_df = classify_students(cleaned_df)
    print("\nFinal Processed Dataset:")
    print(final_df)

    save_data(final_df, args.output)


if __name__ == "__main__":
    main()
