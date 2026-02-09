import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine  # for feature names
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .feature_engineering import process_data
from .utils import setup_logging


def evaluate_model() -> None:
    """Evaluate predicted model performance."""
    setup_logging()
    logging.info("Starting Model Evaluation...")

    output_dir = Path("part2_antigravity/output")
    model_path = output_dir / "model.joblib"

    if not model_path.exists():
        logging.error("Model not found. Run training script first.")
        return

    # Load model
    model = joblib.load(model_path)

    # Load data
    _, X_test, _, y_test, _ = process_data()

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    metrics = {
        "Accuracy": acc,
        "Precision (Weighted)": prec,
        "Recall (Weighted)": rec,
        "F1 Score (Weighted)": f1,
    }

    logging.info("Evaluation Metrics:")
    logging.info(json.dumps(metrics, indent=2))

    # Classification Report
    report = classification_report(y_test, y_pred)
    logging.info("\nClassification Report:\n" + report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    # Feature Importance
    wine_data = load_wine()
    original_feature_names = wine_data.feature_names
    # We added 3 derived features:
    # alcohol_malic_ratio, total_phenols_flavanoids, ash_alcalinity_interaction
    derived_features = [
        "alcohol_malic_ratio",
        "total_phenols_flavanoids",
        "ash_alcalinity_interaction",
    ]
    feature_names = original_feature_names + derived_features

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()

    # Write Report
    with open(output_dir / "evaluation_report.md", "w") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write("## Metrics\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v:.4f}\n")

        f.write("\n## Classification Report\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n")

        f.write("\n## Visualizations\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n")
        f.write("![Feature Importance](feature_importance.png)\n")

        f.write("\n## Recommendations\n")
        f.write("- If accuracy is high, consider deploying checks for data drift.\n")
        f.write(
            "- If certain classes are misclassified, investigate feature "
            "distributions for those classes.\n"
        )
        f.write("- Review feature importance to understand key drivers.\n")

    logging.info("Evaluation completed. Report saved.")


if __name__ == "__main__":
    evaluate_model()
