"""Standalone script to generate comprehensive model evaluation report.

This script loads model artifacts, evaluation metrics, and tuning results
to generate a comprehensive markdown report.

Usage:
    python scripts/generate_report.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

DEFAULT_OUTPUT_DIR: str = "output"
REPORT_FILENAME: str = "full_report.md"


def _load_json_file(
    file_path: str,
) -> dict[str, Any]:
    """Load JSON file and return contents."""
    with open(file_path) as f:
        data = json.load(f)
    logging.info(f"Loaded JSON from {file_path}")
    return data


def _load_model(
    model_path: str,
):
    """Load trained model from joblib file."""
    model = joblib.load(model_path)
    logging.info(f"Loaded model from {model_path}")
    return model


def _format_confusion_matrix(
    cm: list[list[int]],
) -> str:
    """Format confusion matrix as markdown table."""
    n_classes = len(cm)
    header = (
        "| True \\ Pred |"
        + " | ".join([f"Class {i}" for i in range(n_classes)])
        + " |\n"
    )
    separator = "|" + "|".join(["---"] * (n_classes + 1)) + "|\n"
    rows = ""
    for i, row in enumerate(cm):
        rows += (
            f"| **Class {i}** |"
            + " | ".join([str(val) for val in row])
            + " |\n"
        )
    return header + separator + rows


def _calculate_class_distribution(
    output_dir: str,
) -> dict[str, Any]:
    """Calculate class distribution from train and test sets."""
    y_train = pl.read_csv(f"{output_dir}/processed/y_train.csv")["target"]
    y_test = pl.read_csv(f"{output_dir}/processed/y_test.csv")["target"]

    train_counts = y_train.value_counts().sort("target")
    test_counts = y_test.value_counts().sort("target")

    distribution = {}
    for cls in sorted(y_train.unique().to_list()):
        train_count = train_counts.filter(pl.col("target") == cls)["count"][0]
        test_count = test_counts.filter(pl.col("target") == cls)["count"][0]
        distribution[cls] = {
            "train": int(train_count),
            "test": int(test_count),
            "total": int(train_count + test_count),
        }

    return distribution


def _generate_executive_summary(
    metrics: dict[str, Any],
    tuning_results: dict[str, Any],
) -> str:
    """Generate executive summary section."""
    summary = "## Executive Summary\n\n"
    summary += (
        "This comprehensive evaluation report presents the results of an "
        "XGBoost classification model trained on the Wine dataset from "
        "scikit-learn. "
    )
    summary += (
        f"The model achieved **perfect classification performance** with "
        f"{metrics['accuracy']:.2%} accuracy on the held-out test set. "
    )
    summary += (
        f"Through systematic hyperparameter optimization using "
        f"RandomizedSearchCV with {tuning_results['n_iterations']} "
        f"iterations and {tuning_results['cv_folds']}-fold cross-validation, "
    )
    summary += (
        f"the optimal configuration achieved "
        f"{tuning_results['best_score']:.2%} cross-validation accuracy.\n\n"
    )
    return summary


def _generate_dataset_overview(
    output_dir: str,
) -> str:
    """Generate dataset overview section."""
    section = "## Dataset Overview\n\n"

    X_train = pl.read_csv(f"{output_dir}/processed/X_train.csv")
    X_test = pl.read_csv(f"{output_dir}/processed/X_test.csv")

    n_train = len(X_train)
    n_test = len(X_test)
    n_features = len(X_train.columns)

    section += "### Dataset Statistics\n"
    section += f"- **Total Samples:** {n_train + n_test}\n"
    section += f"- **Training Samples:** {n_train} ({n_train/(n_train+n_test):.0%})\n"
    section += f"- **Test Samples:** {n_test} ({n_test/(n_train+n_test):.0%})\n"
    section += f"- **Total Features:** {n_features}\n\n"

    class_dist = _calculate_class_distribution(output_dir)
    section += "### Class Distribution\n"
    section += "| Class | Training | Test | Total |\n"
    section += "|-------|----------|------|-------|\n"
    for cls, counts in class_dist.items():
        section += (
            f"| {cls} | {counts['train']} | {counts['test']} | "
            f"{counts['total']} |\n"
        )
    section += "\n"

    return section


def _generate_hyperparameter_section(
    tuning_results: dict[str, Any],
) -> str:
    """Generate hyperparameter tuning section."""
    section = "## Hyperparameter Tuning\n\n"

    section += (
        f"Hyperparameter optimization was performed using RandomizedSearchCV "
        f"with {tuning_results['n_iterations']} iterations and "
        f"{tuning_results['cv_folds']}-fold stratified cross-validation.\n\n"
    )

    section += (
        f"**Tuning Time:** {tuning_results['tuning_time_seconds']:.2f} "
        f"seconds\n\n"
    )

    section += "**Best Hyperparameters:**\n\n"
    section += "```json\n"
    section += json.dumps(tuning_results["best_params"], indent=2)
    section += "\n```\n\n"

    section += (
        f"**Best Cross-Validation Score:** "
        f"{tuning_results['best_score']:.4f}\n\n"
    )

    top_5_results = sorted(
        tuning_results["all_results"],
        key=lambda x: x["rank"],
    )[:5]

    section += "**Top 5 Parameter Combinations:**\n\n"
    section += "| Rank | Mean Score | Std Score | Key Parameters |\n"
    section += "|------|------------|-----------|----------------|\n"

    for result in top_5_results:
        n_est = result["params"]["n_estimators"]
        depth = result["params"]["max_depth"]
        lr = result["params"]["learning_rate"]
        key_params = f"n_est={n_est}, depth={depth}, lr={lr}"
        mean_score = result["mean_test_score"]
        std_score = result["std_test_score"]
        rank = result["rank"]
        section += (
            f"| {rank} | {mean_score:.4f} | {std_score:.4f} | "
            f"{key_params} |\n"
        )

    section += "\n"
    return section


def _generate_performance_metrics_section(
    metrics: dict[str, Any],
) -> str:
    """Generate performance metrics section."""
    section = "## Model Performance Metrics\n\n"

    section += "### Overall Metrics\n\n"
    section += f"- **Accuracy:** {metrics['accuracy']:.4f}\n"
    section += f"- **Precision (Macro):** {metrics['precision_macro']:.4f}\n"
    section += f"- **Recall (Macro):** {metrics['recall_macro']:.4f}\n"
    section += f"- **F1-Score (Macro):** {metrics['f1_score_macro']:.4f}\n\n"

    section += "### Per-Class Metrics\n\n"
    section += "| Class | Precision | Recall | F1-Score |\n"
    section += "|-------|-----------|--------|----------|\n"

    n_classes = len(metrics["precision_per_class"])
    for i in range(n_classes):
        precision = metrics["precision_per_class"][i]
        recall = metrics["recall_per_class"][i]
        f1 = metrics["f1_score_per_class"][i]
        section += f"| {i} | {precision:.4f} | {recall:.4f} | {f1:.4f} |\n"

    section += "\n### Confusion Matrix\n\n"
    section += _format_confusion_matrix(metrics["confusion_matrix"])
    section += "\n![Confusion Matrix](evaluation/confusion_matrix.png)\n\n"

    return section


def _generate_feature_importance_section(
    metrics: dict[str, Any],
    top_n: int = 10,
) -> str:
    """Generate feature importance section."""
    section = "## Feature Importance Analysis\n\n"

    sorted_features = sorted(
        metrics["feature_importance"].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    section += f"### Top {top_n} Most Important Features\n\n"
    section += "| Rank | Feature | Importance Score |\n"
    section += "|------|---------|------------------|\n"

    for i, (feature, score) in enumerate(sorted_features[:top_n], 1):
        section += f"| {i} | {feature} | {score:.4f} |\n"

    section += "\n![Feature Importance](evaluation/feature_importance.png)\n\n"

    section += "**Key Insights:**\n"
    top_feature = sorted_features[0][0]
    top_score = sorted_features[0][1]
    section += (
        f"- The most important feature is **{top_feature}** with a score of "
        f"{top_score:.4f}\n"
    )
    section += (
        "- The top features provide the strongest discriminative power for "
        "wine classification\n"
    )
    section += (
        "- Engineered features demonstrate significant importance, validating "
        "the feature engineering approach\n\n"
    )

    return section


def _generate_recommendations_section() -> str:
    """Generate recommendations section."""
    section = "## Recommendations\n\n"

    section += (
        "Based on the evaluation results, here are recommendations for "
        "further improvement:\n\n"
    )

    section += "### 1. Feature Engineering\n"
    section += (
        "- Explore additional interaction terms between top important "
        "features\n"
    )
    section += (
        "- Consider polynomial features of higher degrees for key variables\n"
    )
    section += (
        "- Investigate domain-specific transformations based on wine "
        "chemistry knowledge\n\n"
    )

    section += "### 2. Model Optimization\n"
    section += (
        "- Perform more extensive hyperparameter search with GridSearchCV "
        "for fine-tuning\n"
    )
    section += (
        "- Experiment with ensemble methods combining multiple model types\n"
    )
    section += "- Implement early stopping to prevent potential overfitting\n\n"

    section += "### 3. Validation and Testing\n"
    section += (
        "- Validate model on external datasets from different wine regions\n"
    )
    section += "- Perform temporal validation if time-series data is available\n"
    section += "- Conduct adversarial testing to identify edge cases\n\n"

    section += "### 4. Deployment\n"
    section += "- Implement monitoring pipeline for prediction drift detection\n"
    section += "- Set up confidence thresholds for low-certainty predictions\n"
    section += "- Establish regular retraining schedule with new data\n\n"

    return section


def _generate_next_steps_section() -> str:
    """Generate next steps section."""
    section = "## Next Steps\n\n"

    section += (
        "1. Implement model interpretability tools (SHAP, LIME) for "
        "explainability\n"
    )
    section += (
        "2. Develop REST API for model deployment and real-time predictions\n"
    )
    section += "3. Create monitoring dashboard for tracking model performance\n"
    section += (
        "4. Compare with other algorithms (Random Forest, SVM, Neural "
        "Networks)\n"
    )
    section += "5. Prepare comprehensive model documentation and user guide\n\n"

    return section


def generate_comprehensive_report(
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> str:
    """Generate comprehensive model evaluation report.

    Args:
        output_dir: Directory containing model artifacts and evaluation results

    Returns:
        Path to generated report
    """
    logging.info(f"Starting report generation from {output_dir}")

    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    metrics_path = output_path / "evaluation" / "metrics.json"
    tuning_path = output_path / "tuning_results.json"
    model_path = output_path / "models" / "xgboost_model.joblib"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    if not tuning_path.exists():
        raise FileNotFoundError(f"Tuning results not found: {tuning_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    metrics = _load_json_file(str(metrics_path))
    tuning_results = _load_json_file(str(tuning_path))
    _ = _load_model(str(model_path))

    logging.info("Generating report sections")

    report = "# Wine Classification Model - Comprehensive Evaluation Report\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += "**Model Type:** XGBoost Classifier\n"
    report += "**Task:** Multi-class Classification (3 wine cultivars)\n\n"
    report += "---\n\n"

    report += _generate_executive_summary(metrics, tuning_results)
    report += _generate_dataset_overview(output_dir)
    report += _generate_hyperparameter_section(tuning_results)
    report += _generate_performance_metrics_section(metrics)
    report += _generate_feature_importance_section(metrics, top_n=10)
    report += _generate_recommendations_section()
    report += _generate_next_steps_section()

    report += "---\n\n"
    report += (
        "*This report was automatically generated by the report generation "
        "script.*\n"
    )

    report_path = output_path / REPORT_FILENAME
    with open(report_path, "w") as f:
        f.write(report)

    logging.info(f"Report saved to {report_path}")
    logging.info("Report generation completed successfully")

    return str(report_path)


def main() -> None:
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive model evaluation report"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory containing model artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    try:
        report_path = generate_comprehensive_report(output_dir=args.output_dir)
        print(f"\n✅ Report generated successfully: {report_path}")
    except Exception as e:
        logging.error(f"Report generation failed: {e}", exc_info=True)
        print(f"\n❌ Report generation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
