import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from .utils import load_data, setup_logging


def _plot_distributions(df: pl.DataFrame, output_dir: Path) -> None:
    """Generate and save distribution plots for all features."""
    features = [col for col in df.columns if col != "target"]
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png")
    plt.close()


def _plot_correlation_heatmap(df: pl.DataFrame, output_dir: Path) -> None:
    """Generate and save correlation heatmap."""
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()


def _check_class_balance(df: pl.DataFrame) -> None:
    """Log class distribution."""
    class_counts = df["target"].value_counts()
    logging.info("Class Balance:")
    logging.info(json.dumps(class_counts.to_dicts(), indent=2))


def _detect_outliers(df: pl.DataFrame) -> None:
    """Log simple outlier detection based on IQR for demonstration."""
    features = [col for col in df.columns if col != "target"]
    outlier_summary = {}

    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df.filter((pl.col(feature) < lower_bound) | (pl.col(feature) > upper_bound))
        outlier_summary[feature] = len(outliers)

    logging.info("Outlier Summary (Number of outliers per feature):")
    logging.info(json.dumps(outlier_summary, indent=2))


def run_eda() -> None:
    """Run the full EDA pipeline."""
    setup_logging()
    logging.info("Starting EDA...")

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # Summary Statistics
    logging.info("Data Summary:")
    summary = df.describe()
    logging.info(json.dumps(summary.to_dicts(), indent=2))

    _plot_distributions(df, output_dir)
    _plot_correlation_heatmap(df, output_dir)
    _check_class_balance(df)
    _detect_outliers(df)

    logging.info("EDA completed successfully.")


if __name__ == "__main__":
    run_eda()
