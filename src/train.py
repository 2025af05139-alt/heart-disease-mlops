"""Train and compare reproducible heart disease classification pipelines."""

from __future__ import annotations

import json
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "Heart Disease Classification"

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed CSV and validate its required columns."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run 'python -m src.data' first."
        )

    df = pd.read_csv(DATA_PATH)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    if not y.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Create leakage-safe preprocessing for numeric and categorical data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, BaseEstimator]:
    """Define two reproducible baseline classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Bundle preprocessing and classification into one estimator."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("classifier", model),
        ]
    )


def plot_model_comparison(results: pd.DataFrame) -> None:
    """Save a visual comparison of held-out test metrics."""
    columns = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_roc_auc",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    results.set_index("model")[columns].plot(
        kind="bar",
        ylim=(0, 1),
        ax=ax,
    )
    ax.set_title("Model comparison on the held-out test set")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(
        ["Accuracy", "Precision", "Recall", "ROC-AUC"],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "model_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_roc_curves(
    actual: pd.Series,
    test_scores: dict[str, np.ndarray],
) -> None:
    """Save ROC curves using the held-out test predictions."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, scores in test_scores.items():
        fpr, tpr, _ = roc_curve(actual, scores)
        auc = roc_auc_score(actual, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set(
        title="ROC curves on the held-out test set",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "roc_curves.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def scalar_model_params(model: BaseEstimator) -> dict[str, Any]:
    """Return simple model parameters that MLflow can log as parameters."""
    params: dict[str, Any] = {}
    for key, value in model.get_params(deep=True).items():
        if isinstance(value, (str, int, float, bool)):
            params[f"classifier.{key}"] = value
    return params


def main() -> None:
    """Train, evaluate, track, compare, and save the selected model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Use a project-local SQLite database (not the deprecated filesystem store).
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X, y = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    test_scores: dict[str, np.ndarray] = {}
    run_ids: dict[str, str] = {}

    for name, model in build_models().items():
        pipeline = build_pipeline(model)

        # Each model gets a separate MLflow run.
        with mlflow.start_run(run_name=name) as active_run:
            run_ids[name] = active_run.info.run_id
            mlflow.set_tags(
                {
                    "model_name": name,
                    "dataset": "UCI Heart Disease Cleveland subset",
                    "selection_metric": "mean CV ROC-AUC",
                    "random_state": str(RANDOM_STATE),
                }
            )

            params: dict[str, Any] = {
                "model_name": name,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "cv_folds": CV_FOLDS,
                "preprocessing.numeric_columns": ",".join(NUMERIC_COLUMNS),
                "preprocessing.categorical_columns": ",".join(CATEGORICAL_COLUMNS),
                "preprocessing.numeric_imputer": "median",
                "preprocessing.categorical_imputer": "most_frequent",
                "preprocessing.scaler": "StandardScaler",
                "preprocessing.encoder": "OneHotEncoder",
            }
            params.update(scalar_model_params(model))
            mlflow.log_params(params)

            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=SCORING,
                n_jobs=1,
                error_score="raise",
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            row: dict[str, float | str] = {"model": name}
            metrics_to_log: dict[str, float] = {}

            for metric in SCORING:
                values = cv_results[f"test_{metric}"]
                mean_score = float(values.mean())
                std_score = float(values.std(ddof=1))
                row[f"cv_{metric}_mean"] = mean_score
                row[f"cv_{metric}_std"] = std_score
                metrics_to_log[f"cv_{metric}_mean"] = mean_score
                metrics_to_log[f"cv_{metric}_std"] = std_score

                for fold_number, fold_score in enumerate(values, start=1):
                    metrics_to_log[f"cv_{metric}_fold_{fold_number}"] = float(fold_score)

            test_metrics = {
                "test_accuracy": float(accuracy_score(y_test, predictions)),
                "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
                "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            }
            row.update(test_metrics)
            metrics_to_log.update(test_metrics)
            mlflow.log_metrics(metrics_to_log)

            # Add an input schema/example to make the logged model easier to reuse.
            input_example = X_train.dropna().head(3)
            if input_example.empty:
                input_example = X_train.head(3)
            signature = infer_signature(
                input_example,
                pipeline.predict(input_example),
            )
            # This pipeline was trained by this project from local code and data.
            # Use cloudpickle for MLflow logging to avoid skops type-audit failures
            # for NumPy/scikit-learn internals. Load only model artifacts you trust.
            log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
            )

            rows.append(row)
            fitted_models[name] = pipeline
            test_scores[name] = probabilities

    results = pd.DataFrame(rows)

    # Select using CV ROC-AUC—not the held-out test metrics.
    best_row = max(rows, key=lambda row: float(row["cv_roc_auc_mean"]))
    best_name = str(best_row["model"])
    model_path = MODEL_DIR / "heart_disease_best_model.joblib"
    joblib.dump(fitted_models[best_name], model_path)

    comparison_path = REPORT_DIR / "model_comparison.csv"
    results.to_csv(comparison_path, index=False)
    plot_model_comparison(results)
    plot_roc_curves(y_test, test_scores)

    summary = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "dataset_rows": len(X),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "selected_model": best_name,
        "selection_metric": "mean cross-validation ROC-AUC",
        "model_artifact": str(model_path.relative_to(PROJECT_ROOT)),
        "mlflow_experiment": MLFLOW_EXPERIMENT_NAME,
        "results": rows,
    }

    metrics_path = REPORT_DIR / "model_metrics.json"
    metrics_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Attach cross-model comparison artifacts to both model runs.
    client = MlflowClient()
    artifacts_to_log = [
        comparison_path,
        metrics_path,
        FIGURE_DIR / "model_comparison.png",
        FIGURE_DIR / "roc_curves.png",
    ]

    for name, run_id in run_ids.items():
        client.set_tag(
            run_id,
            "selected_by_cv_roc_auc",
            str(name == best_name).lower(),
        )
        for artifact_path in artifacts_to_log:
            if artifact_path.exists():
                client.log_artifact(
                    run_id,
                    str(artifact_path),
                    artifact_path="evaluation",
                )

    # Keep an easy-to-find copy of the chosen portable model in its selected run.
    client.log_artifact(
        run_ids[best_name],
        str(model_path),
        artifact_path="selected_model",
    )

    print(f"Dataset rows: {len(X)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Cross-validation folds: {CV_FOLDS}")
    print("\nModel comparison:")
    print(
        results[
            [
                "model",
                "cv_roc_auc_mean",
                "cv_roc_auc_std",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_roc_auc",
            ]
        ].to_string(index=False)
    )
    print(f"\nSelected model by CV ROC-AUC: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved comparison plot: {FIGURE_DIR / 'model_comparison.png'}")
    print(f"Saved ROC curves: {FIGURE_DIR / 'roc_curves.png'}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"MLflow tracking database: {MLFLOW_DB_PATH}")
    print("MLflow run IDs:")
    for name, run_id in run_ids.items():
        print(f"  {name}: {run_id}")


if __name__ == "__main__":
    main()
