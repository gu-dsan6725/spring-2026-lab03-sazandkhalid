import json
import logging

import polars as pl
from sklearn.datasets import load_wine


def setup_logging() -> None:
    """Correctly setup logging with the required format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
    )


def log_data_info(data: dict) -> None:
    """Log dictionary data in a pretty format."""
    logging.info(json.dumps(data, indent=2, default=str))


def load_data() -> pl.DataFrame:
    """Load the Wine dataset and convert to Polars DataFrame."""
    wine = load_wine()
    data = pl.DataFrame(wine.data, schema=wine.feature_names)
    data = data.with_columns(pl.Series("target", wine.target))
    return data
