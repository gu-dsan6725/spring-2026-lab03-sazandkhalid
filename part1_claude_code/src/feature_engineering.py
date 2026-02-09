"""Feature engineering for Wine dataset."""

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: str = "output/processed"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42


def _create_derived_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create derived features from original features."""
    df = df.with_columns([
        (pl.col("alcohol") / pl.col("malic_acid")).alias("alcohol_malic_ratio"),
        (
            pl.col("flavanoids") * pl.col("od280/od315_of_diluted_wines")
        ).alias("flavanoid_od_interaction"),
        (pl.col("total_phenols") ** 2).alias("total_phenols_squared"),
    ])
    feature_list = (
        "alcohol_malic_ratio, flavanoid_od_interaction, total_phenols_squared"
    )
    logging.info(f"Created 3 derived features: {feature_list}")
    return df


def _apply_standard_scaling(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Apply standard scaling to features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info("Applied StandardScaler to all features")
    return X_train_scaled, X_test_scaled, scaler


def _save_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save processed data to CSV files."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    pl.DataFrame(X_train, schema=feature_names).write_csv(f"{OUTPUT_DIR}/X_train.csv")
    pl.DataFrame(X_test, schema=feature_names).write_csv(f"{OUTPUT_DIR}/X_test.csv")
    pl.DataFrame({"target": y_train}).write_csv(f"{OUTPUT_DIR}/y_train.csv")
    pl.DataFrame({"target": y_test}).write_csv(f"{OUTPUT_DIR}/y_test.csv")

    logging.info(f"Saved processed data to {OUTPUT_DIR}/")


def engineer_features() -> dict[str, any]:
    """Engineer features for Wine dataset."""
    logging.info("Loading Wine dataset")
    wine_data = load_wine()
    feature_names = list(wine_data.feature_names)

    df = pl.DataFrame(
        data=wine_data.data,
        schema=feature_names,
    ).with_columns(pl.Series("target", wine_data.target))

    logging.info(f"Original dataset shape: {df.shape}")

    df = _create_derived_features(df)

    all_feature_names = [col for col in df.columns if col != "target"]
    logging.info(f"Total features after engineering: {len(all_feature_names)}")

    X = df.select(all_feature_names).to_numpy()
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logging.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    X_train_scaled, X_test_scaled, scaler = _apply_standard_scaling(X_train, X_test)

    _save_data(X_train_scaled, X_test_scaled, y_train, y_test, all_feature_names)

    metadata = {
        "n_features": len(all_feature_names),
        "feature_names": all_feature_names,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_split_ratio": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }

    logging.info(f"Feature engineering metadata:\n{json.dumps(metadata, indent=2)}")
    logging.info("Feature engineering completed successfully")

    return metadata


if __name__ == "__main__":
    engineer_features()
