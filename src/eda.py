"""Generate reproducible exploratory-data-analysis summaries and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Works in headless CI/container environments.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import (
    PROJECT_ROOT,
    TARGET_COLUMN,
    clean_data,
    download_raw_data,
    load_raw_data,
    save_clean_data,
)

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


def run_eda(df: pd.DataFrame, output_dir: str | Path = FIGURE_DIR) -> dict[str, object]:
    """Save EDA figures and return machine-readable dataset diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    numeric = df.select_dtypes(include="number")
    missing = df.isna().sum().sort_values(ascending=False)
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {str(k): int(v) for k, v in missing.items() if v > 0},
        "target_counts": {
            str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "describe": df.describe(include="all").fillna("").to_dict(),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET_COLUMN, ax=ax)
    ax.set(
        title="Target class balance",
        xlabel="Target (0 = no disease, 1 = disease)",
        ylabel="Number of records",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "class_balance.png", dpi=160)
    plt.close(fig)

    numeric.drop(columns=[TARGET_COLUMN], errors="ignore").hist(
        figsize=(14, 10), bins=20, edgecolor="black"
    )
    plt.suptitle("Feature distributions", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_histograms.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    # Correlations are exploratory only; categorical codes are not necessarily ordinal.
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(numeric.corr(numeric_only=True), cmap="vlag", center=0, annot=False, ax=ax)
    ax.set_title("Numeric correlation heatmap (encoded categories require caution)")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    (output_dir / "eda_summary.json").write_text(
        __import__("json").dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main() -> None:
    raw_path = download_raw_data()
    raw_df = load_raw_data(raw_path)
    cleaned = clean_data(raw_df)
    clean_path = save_clean_data(cleaned)
    summary = run_eda(cleaned)
    print(f"Saved cleaned dataset: {clean_path}")
    print(f"Saved EDA artifacts to: {FIGURE_DIR}")
    print(
        f"Rows={summary['rows']}, "
        f"columns={summary['columns']}, "
        f"duplicates={summary['duplicate_rows']}"
    )
    print(f"Missing values: {summary['missing_values']}")
    print(f"Target counts: {summary['target_counts']}")


if __name__ == "__main__":
    main()
