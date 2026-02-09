"""Exploratory Data Analysis for Wine dataset."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output/eda"
FIGURE_SIZE: tuple[int, int] = (12, 10)
IQR_MULTIPLIER: float = 1.5


def _detect_outliers_iqr(
    df: pl.DataFrame,
    column: str,
) -> dict[str, int | float]:
    """Detect outliers using IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - IQR_MULTIPLIER * iqr
    upper_bound = q3 + IQR_MULTIPLIER * iqr
    outliers = df.filter(
        (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
    )
    return {
        "feature": column,
        "n_outliers": len(outliers),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


def _create_distribution_plots(
    df: pl.DataFrame,
    feature_names: list[str],
) -> None:
    """Create distribution plots for all features."""
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_names):
        axes[idx].hist(df[feature].to_numpy(), bins=30, edgecolor="black", alpha=0.7)
        axes[idx].set_title(f"Distribution of {feature}", fontsize=10)
        axes[idx].set_xlabel(feature, fontsize=9)
        axes[idx].set_ylabel("Frequency", fontsize=9)
        axes[idx].grid(alpha=0.3)

    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Distribution plots saved")


def _create_correlation_heatmap(
    df: pl.DataFrame,
    feature_names: list[str],
) -> None:
    """Create correlation heatmap."""
    corr_matrix = df.select(feature_names).to_pandas().corr()

    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Correlation heatmap saved")


def _create_class_balance_plot(
    df: pl.DataFrame,
) -> dict[str, int]:
    """Create class balance plot and return counts."""
    class_counts = df.group_by("target").agg(pl.count()).sort("target")

    plt.figure(figsize=(8, 6))
    plt.bar(
        class_counts["target"].to_numpy(),
        class_counts["count"].to_numpy(),
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("Wine Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Class Balance in Wine Dataset", fontsize=14)
    plt.xticks(class_counts["target"].to_numpy())
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_balance.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Class balance plot saved")

    return {int(row[0]): int(row[1]) for row in class_counts.iter_rows()}


def perform_eda() -> None:
    """Perform exploratory data analysis on Wine dataset."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    logging.info("Loading Wine dataset")
    wine_data = load_wine()
    feature_names = wine_data.feature_names

    df = pl.DataFrame(
        data=wine_data.data,
        schema=feature_names,
    ).with_columns(pl.Series("target", wine_data.target))

    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Number of features: {len(feature_names)}")
    logging.info(f"Number of classes: {df['target'].n_unique()}")

    summary_stats = df.select(feature_names).describe()
    logging.info(f"Summary statistics:\n{summary_stats}")

    _create_distribution_plots(df, feature_names)

    _create_correlation_heatmap(df, feature_names)

    class_counts = _create_class_balance_plot(df)
    logging.info(f"Class distribution: {json.dumps(class_counts, indent=2)}")

    logging.info("Detecting outliers using IQR method")
    outlier_results = []
    for feature in feature_names:
        outlier_info = _detect_outliers_iqr(df, feature)
        outlier_results.append(outlier_info)

    outlier_json = json.dumps(outlier_results, indent=2, default=str)
    logging.info(f"Outlier detection results:\n{outlier_json}")

    outlier_counts = [r["n_outliers"] for r in outlier_results]
    feature_labels = [r["feature"] for r in outlier_results]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_labels)), outlier_counts, edgecolor="black", alpha=0.7)
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Number of Outliers", fontsize=12)
    plt.title("Outlier Detection Summary (IQR Method)", fontsize=14)
    plt.xticks(range(len(feature_labels)), feature_labels, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/outliers_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Outlier summary plot saved")

    logging.info("EDA completed successfully")


if __name__ == "__main__":
    perform_eda()
