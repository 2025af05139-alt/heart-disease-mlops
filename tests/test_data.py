"""Unit tests for data cleaning and validation."""
import pandas as pd
import pytest

from src.data import FEATURE_NAMES, clean_data, load_raw_data


def sample_raw_frame() -> pd.DataFrame:
    return pd.DataFrame([
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, "?", 3, 2],
    ], columns=FEATURE_NAMES)


def test_clean_data_maps_original_target_to_binary() -> None:
    result = clean_data(sample_raw_frame())
    assert result["target"].tolist() == [0, 1]
    assert set(result["target"].unique()).issubset({0, 1})
    assert "target_raw" not in result.columns


def test_clean_data_preserves_missing_predictors_for_pipeline_imputation() -> None:
    result = clean_data(sample_raw_frame())
    assert pd.isna(result.loc[1, "ca"])


def test_clean_data_requires_expected_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_data(pd.DataFrame({"age": [60]}))


def test_load_raw_data_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "does-not-exist.data")
