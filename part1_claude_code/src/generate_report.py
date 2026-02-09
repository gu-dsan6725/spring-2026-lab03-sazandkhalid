"""Generate comprehensive evaluation report."""

import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

REPORT_PATH: str = "output/report.md"
TOP_N_FEATURES: int = 10


def _load_metrics() -> dict:
    """Load evaluation metrics from JSON file."""
    with open("output/evaluation/metrics.json") as f:
        metrics = json.load(f)
    logging.info("Loaded evaluation metrics")
    return metrics


def _load_tuning_results() -> dict:
    """Load hyperparameter tuning results from JSON file."""
    with open("output/tuning_results.json") as f:
        tuning_results = json.load(f)
    logging.info("Loaded tuning results")
    return tuning_results


def _format_confusion_matrix(
    cm: list[list[int]],
) -> str:
    """Format confusion matrix as markdown table."""
    n_classes = len(cm)
    header = "| True \\ Pred |" + " | ".join([f"Class {i}" for i in range(n_classes)]) + " |\n"
    separator = "|" + "|".join(["---"] * (n_classes + 1)) + "|\n"
    rows = ""
    for i, row in enumerate(cm):
        rows += f"| **Class {i}** |" + " | ".join([str(val) for val in row]) + " |\n"
    return header + separator + rows


def _generate_executive_summary(
    metrics: dict,
    tuning_results: dict,
) -> str:
    """Generate executive summary section."""
    summary = "## Executive Summary\n\n"
    summary += (
        "This report presents the evaluation results of an XGBoost "
        "classification model trained on the Wine dataset. "
    )
    summary += (
        f"The model achieved an accuracy of "
        f"**{metrics['accuracy']:.2%}** on the test set.\n\n"
    )
    summary += "**Key Highlights:**\n"
    summary += "- Model Type: XGBoost Classifier\n"
    n_iters = tuning_results['n_iterations']
    summary += f"- Hyperparameter Tuning: RandomizedSearchCV with {n_iters} iterations\n"
    summary += f"- Best CV Score: {tuning_results['best_score']:.4f}\n"
    summary += f"- Test Accuracy: {metrics['accuracy']:.4f}\n"
    summary += f"- Macro F1-Score: {metrics['f1_score_macro']:.4f}\n\n"
    return summary


def _generate_dataset_overview() -> str:
    """Generate dataset overview section."""
    overview = "## Dataset Overview\n\n"
    overview += (
        "The Wine dataset contains chemical analysis results of wines "
        "grown in the same region in Italy but derived from three "
        "different cultivars.\n\n"
    )
    overview += "**Dataset Characteristics:**\n"
    overview += "- Source: scikit-learn datasets\n"
    overview += "- Task: Multi-class classification (3 classes)\n"
    overview += "- Features: 13 chemical properties + 3 engineered features\n"
    overview += "- Train/Test Split: 80/20 with stratification\n"
    overview += "- Preprocessing: StandardScaler applied to all features\n\n"
    return overview


def _generate_tuning_section(
    tuning_results: dict,
) -> str:
    """Generate hyperparameter tuning section."""
    section = "## Hyperparameter Tuning\n\n"
    n_iters = tuning_results['n_iterations']
    cv_folds = tuning_results['cv_folds']
    section += (
        f"Hyperparameter optimization was performed using "
        f"RandomizedSearchCV with {n_iters} iterations "
    )
    section += f"and {cv_folds}-fold stratified cross-validation.\n\n"
    section += f"**Tuning Time:** {tuning_results['tuning_time_seconds']:.2f} seconds\n\n"
    section += "**Best Hyperparameters:**\n\n"
    section += "```json\n"
    section += json.dumps(tuning_results['best_params'], indent=2)
    section += "\n```\n\n"
    section += f"**Best Cross-Validation Score:** {tuning_results['best_score']:.4f}\n\n"

    top_5_results = sorted(
        tuning_results['all_results'],
        key=lambda x: x['rank'],
    )[:5]

    section += "**Top 5 Parameter Combinations:**\n\n"
    section += "| Rank | Mean Score | Std Score | Key Parameters |\n"
    section += "|------|------------|-----------|----------------|\n"
    for result in top_5_results:
        n_est = result['params']['n_estimators']
        depth = result['params']['max_depth']
        lr = result['params']['learning_rate']
        key_params = f"n_est={n_est}, depth={depth}, lr={lr}"
        mean_score = result['mean_test_score']
        std_score = result['std_test_score']
        rank = result['rank']
        section += f"| {rank} | {mean_score:.4f} | {std_score:.4f} | {key_params} |\n"

    section += "\n"
    return section


def _generate_metrics_section(
    metrics: dict,
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
    for i in range(len(metrics['precision_per_class'])):
        section += f"| {i} | {metrics['precision_per_class'][i]:.4f} | "
        section += f"{metrics['recall_per_class'][i]:.4f} | "
        section += f"{metrics['f1_score_per_class'][i]:.4f} |\n"

    section += "\n### Confusion Matrix\n\n"
    section += _format_confusion_matrix(metrics['confusion_matrix'])
    section += "\n![Confusion Matrix](evaluation/confusion_matrix.png)\n\n"

    return section


def _generate_feature_importance_section(
    metrics: dict,
) -> str:
    """Generate feature importance section."""
    section = "## Feature Importance Analysis\n\n"

    sorted_features = sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True,
    )

    section += f"### Top {TOP_N_FEATURES} Most Important Features\n\n"
    section += "| Rank | Feature | Importance Score |\n"
    section += "|------|---------|------------------|\n"
    for i, (feature, score) in enumerate(sorted_features[:TOP_N_FEATURES], 1):
        section += f"| {i} | {feature} | {score:.4f} |\n"

    section += "\n![Feature Importance](evaluation/feature_importance.png)\n\n"

    section += "**Key Insights:**\n"
    top_feature = sorted_features[0][0]
    top_score = sorted_features[0][1]
    section += (
        f"- The most important feature is **{top_feature}** "
        f"with a score of {top_score:.4f}\n"
    )
    section += (
        "- The top 3 features account for a significant portion of "
        "the model's predictive power\n"
    )
    section += (
        "- Engineered features show varying levels of importance, "
        "validating the feature engineering process\n\n"
    )

    return section


def _generate_recommendations() -> str:
    """Generate recommendations section."""
    recommendations = "## Recommendations\n\n"
    recommendations += (
        "Based on the evaluation results, here are recommendations "
        "for further improvement:\n\n"
    )
    recommendations += "1. **Feature Engineering:**\n"
    recommendations += (
        "   - Explore additional interaction terms between top "
        "important features\n"
    )
    recommendations += (
        "   - Consider polynomial features of higher degrees for "
        "key variables\n"
    )
    recommendations += (
        "   - Investigate domain-specific transformations based on "
        "chemistry knowledge\n\n"
    )
    recommendations += "2. **Model Optimization:**\n"
    recommendations += (
        "   - Perform more extensive hyperparameter search with "
        "GridSearchCV for fine-tuning\n"
    )
    recommendations += (
        "   - Experiment with different objective functions if "
        "class imbalance exists\n"
    )
    recommendations += "   - Consider ensemble methods combining multiple models\n\n"
    recommendations += "3. **Data Collection:**\n"
    recommendations += "   - If possible, collect more samples to improve model robustness\n"
    recommendations += "   - Investigate feature correlations to reduce redundancy\n"
    recommendations += "   - Consider feature selection techniques to simplify the model\n\n"
    recommendations += "4. **Deployment Considerations:**\n"
    recommendations += "   - Monitor model performance on new data\n"
    recommendations += "   - Implement confidence thresholds for predictions\n"
    recommendations += "   - Set up regular retraining pipelines\n\n"
    return recommendations


def _generate_next_steps() -> str:
    """Generate next steps section."""
    next_steps = "## Next Steps\n\n"
    next_steps += "1. Validate model on additional hold-out datasets if available\n"
    next_steps += "2. Perform error analysis on misclassified samples\n"
    next_steps += "3. Compare with other algorithms (Random Forest, SVM, Neural Networks)\n"
    next_steps += "4. Implement model interpretability techniques (SHAP values, LIME)\n"
    next_steps += "5. Prepare model for production deployment\n\n"
    return next_steps


def generate_report() -> None:
    """Generate comprehensive evaluation report."""
    logging.info("Generating evaluation report")

    metrics = _load_metrics()
    tuning_results = _load_tuning_results()

    report = "# Wine Classification Model Evaluation Report\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += "---\n\n"

    report += _generate_executive_summary(metrics, tuning_results)
    report += _generate_dataset_overview()
    report += _generate_tuning_section(tuning_results)
    report += _generate_metrics_section(metrics)
    report += _generate_feature_importance_section(metrics)
    report += _generate_recommendations()
    report += _generate_next_steps()

    report += "---\n\n"
    report += "*This report was automatically generated by the Wine Classification Pipeline.*\n"

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    logging.info(f"Report saved to {REPORT_PATH}")
    logging.info("Report generation completed successfully")


if __name__ == "__main__":
    generate_report()
