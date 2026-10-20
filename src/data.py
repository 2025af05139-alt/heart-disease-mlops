"""Download, validate, and clean the UCI Heart Disease Cleveland dataset."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
RAW_FILENAME = "processed.cleveland.data"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target_raw",
]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"


def download_raw_data(force: bool = False, timeout: int = 30) -> Path:
    """Download the official UCI archive and extract the Cleveland data file.

    The downloaded archive is retained in data/raw for provenance. Existing data
    is reused unless force=True.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / RAW_FILENAME
    zip_path = RAW_DIR / "heart-disease-uci.zip"
    if raw_path.exists() and not force:
        return raw_path

    response = requests.get(ZIP_URL, timeout=timeout)
    response.raise_for_status()
    if not response.content.startswith(b"PK"):
        raise ValueError("UCI download did not look like a ZIP archive.")
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        matches = [name for name in archive.namelist() if name.lower().endswith(RAW_FILENAME)]
        if not matches:
            raise FileNotFoundError(f"Could not find {RAW_FILENAME} in the UCI archive.")
        raw_path.write_bytes(archive.read(matches[0]))
    return raw_path


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read Cleveland data and convert UCI '?' missing-value markers to NA."""
    data_path = Path(path) if path is not None else RAW_DIR / RAW_FILENAME
    if not data_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {data_path}. Run: python -m src.data"
        )
    df = pd.read_csv(data_path, header=None, names=FEATURE_NAMES, na_values=["?"])
    if df.shape[1] != 14:
        raise ValueError(f"Expected 14 columns, received {df.shape[1]}.")
    for column in FEATURE_NAMES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the target to 0/1 and leave missing predictors for pipeline imputation."""
    required = set(FEATURE_NAMES)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df.copy()
    # Be defensive when callers pass a DataFrame directly rather than load_raw_data().
    for column in FEATURE_NAMES:
        cleaned[column] = pd.to_numeric(cleaned[column].replace("?", pd.NA), errors="coerce")
    # Original UCI target: 0 = no disease; 1–4 = presence of disease.
    cleaned[TARGET_COLUMN] = (cleaned["target_raw"] > 0).astype("int64")
    cleaned = cleaned.drop(columns=["target_raw"])
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    if not cleaned[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Target must contain only binary labels 0 and 1.")
    return cleaned


def save_clean_data(df: pd.DataFrame, output_path: str | Path | None = None) -> Path:
    """Save the cleaned dataset as CSV, preserving missing predictors as empty cells."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    destination = Path(output_path) if output_path else PROCESSED_DIR / "heart_disease_clean.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination


def main() -> None:
    path = download_raw_data()
    raw = load_raw_data(path)
    cleaned = clean_data(raw)
    output = save_clean_data(cleaned)
    print(f"Raw dataset: {path}")
    print(f"Clean dataset: {output}")
    print(f"Rows, columns: {cleaned.shape}")
    print("Missing values per column:")
    print(cleaned.isna().sum().to_string())
    print("Target counts (0=no disease, 1=disease):")
    print(cleaned[TARGET_COLUMN].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
