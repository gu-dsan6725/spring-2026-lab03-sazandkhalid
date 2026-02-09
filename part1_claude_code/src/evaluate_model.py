"""Evaluate trained XGBoost model."""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output/evaluation"
METRICS_FILE: str = "metrics.json"


def _load_model_and_data() -> tuple:
    """Load trained model and test data."""
    model = joblib.load("output/models/xgboost_model.joblib")
    X_test = pl.read_csv("output/processed/X_test.csv").to_numpy()
    y_test = pl.read_csv("output/processed/y_test.csv")["target"].to_numpy()
    logging.info(f"Loaded model and test data: X_test shape {X_test.shape}")
    return model, X_test, y_test


def _calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_score_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_per_class": precision_score(y_true, y_pred, average=None).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None).tolist(),
        "f1_score_per_class": f1_score(y_true, y_pred, average=None).tolist(),
    }

    class_report = classification_report(y_true, y_pred, output_dict=True)
    metrics["classification_report"] = class_report

    logging.info(f"Evaluation metrics:\n{json.dumps(metrics, indent=2, default=str)}")

    return metrics


def _create_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=14, pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Confusion matrix plot saved")

    return cm


def _create_feature_importance_plot(
    model,
    feature_names: list[str],
) -> dict:
    """Create and save feature importance plot."""
    importance_scores = model.feature_importances_
    importance_dict = {
        name: float(score)
        for name, score in zip(feature_names, importance_scores)
    }

    sorted_importance = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    top_n = min(15, len(sorted_importance))
    top_features = sorted_importance[:top_n]

    features = [f[0] for f in top_features]
    scores = [f[1] for f in top_features]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), scores, edgecolor="black", alpha=0.7)
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("Top Feature Importances", fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Feature importance plot saved")

    return importance_dict


def _save_metrics(
    metrics: dict,
) -> None:
    """Save metrics to JSON file."""
    metrics_path = f"{OUTPUT_DIR}/{METRICS_FILE}"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logging.info(f"Metrics saved to {metrics_path}")


def evaluate_model() -> dict:
    """Evaluate trained model on test set."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model, X_test, y_test = _load_model_and_data()

    logging.info("Generating predictions on test set")
    y_pred = model.predict(X_test)

    metrics = _calculate_metrics(y_test, y_pred)

    cm = _create_confusion_matrix_plot(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    feature_names = pl.read_csv("output/processed/X_test.csv").columns
    importance_dict = _create_feature_importance_plot(model, feature_names)
    metrics["feature_importance"] = importance_dict

    _save_metrics(metrics)

    logging.info("Model evaluation completed successfully")

    return metrics


if __name__ == "__main__":
    evaluate_model()
