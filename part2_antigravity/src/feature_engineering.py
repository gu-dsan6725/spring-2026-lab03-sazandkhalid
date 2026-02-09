import logging  # Added import

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import load_data, setup_logging


def _create_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create derived features."""
    # Example derived features
    # 1. Alcohol / Malic Acid Ratio (interaction)
    # 2. Total Phenols * Flavanoids (interaction)
    # 3. Ash * Alcalinity of Ash (interaction)

    df = df.with_columns(
        [
            (pl.col("alcohol") / pl.col("malic_acid")).alias("alcohol_malic_ratio"),
            (pl.col("total_phenols") * pl.col("flavanoids")).alias("total_phenols_flavanoids"),
            (pl.col("ash") * pl.col("alcalinity_of_ash")).alias("ash_alcalinity_interaction"),
        ]
    )
    return df


def process_data(test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Load data, create features, scale, and split.
    Returns: X_train, X_test, y_train, y_test (as numpy arrays/pandas dfs for sklearn compatibility)
    """
    setup_logging()
    logging.info("Starting Feature Engineering...")

    df = load_data()
    df = _create_derived_features(df)

    target = df["target"].to_numpy()
    features = df.drop("target").to_pandas()  # XGBoost/Sklearn work best with pandas/numpy

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, stratify=target, random_state=random_state
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for convenience if needed, but numpy/pandas is fine for XGBoost
    # Keeping them as numpy arrays for the model

    logging.info(
        f"Data processed. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}"
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    process_data()
