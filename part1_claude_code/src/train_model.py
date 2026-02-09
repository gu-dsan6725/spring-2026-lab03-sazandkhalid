"""Train XGBoost model with hyperparameter tuning."""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

MODEL_DIR: str = "output/models"
TUNING_RESULTS_PATH: str = "output/tuning_results.json"
CV_FOLDS: int = 5
N_ITER_SEARCH: int = 20
RANDOM_STATE: int = 42

PARAM_DISTRIBUTIONS: dict[str, list] = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3, 0.4],
}


def _load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load training data from CSV files."""
    X_train = pl.read_csv("output/processed/X_train.csv").to_numpy()
    y_train = pl.read_csv("output/processed/y_train.csv")["target"].to_numpy()
    logging.info(
        f"Loaded training data: X_train shape {X_train.shape}, "
        f"y_train shape {y_train.shape}"
    )
    return X_train, y_train


def _perform_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[dict, RandomizedSearchCV]:
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    logging.info("Starting hyperparameter tuning with RandomizedSearchCV")
    logging.info(f"Search space: {json.dumps(PARAM_DISTRIBUTIONS, indent=2)}")

    base_model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    start_time = time.time()

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    random_search.fit(X_train, y_train)

    tuning_time = time.time() - start_time

    tuning_results = {
        "best_params": random_search.best_params_,
        "best_score": float(random_search.best_score_),
        "n_iterations": N_ITER_SEARCH,
        "cv_folds": CV_FOLDS,
        "tuning_time_seconds": float(tuning_time),
        "all_results": [
            {
                "params": {
                    k: (
                        int(v)
                        if isinstance(v, (np.integer, np.int64))
                        else float(v)
                        if isinstance(v, (np.floating, np.float64))
                        else v
                    )
                    for k, v in random_search.cv_results_["params"][i].items()
                },
                "mean_test_score": float(
                    random_search.cv_results_["mean_test_score"][i]
                ),
                "std_test_score": float(
                    random_search.cv_results_["std_test_score"][i]
                ),
                "rank": int(random_search.cv_results_["rank_test_score"][i]),
            }
            for i in range(len(random_search.cv_results_["params"]))
        ],
    }

    logging.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    logging.info(f"Best score: {random_search.best_score_:.4f}")
    logging.info(f"Best parameters: {json.dumps(tuning_results['best_params'], indent=2)}")

    return tuning_results, random_search


def _save_tuning_results(
    tuning_results: dict,
) -> None:
    """Save tuning results to JSON file."""
    with open(TUNING_RESULTS_PATH, "w") as f:
        json.dump(tuning_results, f, indent=2)
    logging.info(f"Tuning results saved to {TUNING_RESULTS_PATH}")


def _train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
) -> XGBClassifier:
    """Train final model with best hyperparameters."""
    logging.info("Training final model with best hyperparameters")

    final_model = XGBClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )

    final_model.fit(X_train, y_train)
    logging.info("Final model training completed")

    return final_model


def _evaluate_with_cv(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Evaluate model with cross-validation."""
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
    )

    cv_results = {
        "cv_scores": [float(score) for score in cv_scores],
        "mean_cv_score": float(np.mean(cv_scores)),
        "std_cv_score": float(np.std(cv_scores)),
    }

    logging.info(f"Cross-validation results: {json.dumps(cv_results, indent=2)}")

    return cv_results


def _save_model(
    model: XGBClassifier,
) -> None:
    """Save trained model to file."""
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model_path = f"{MODEL_DIR}/xgboost_model.joblib"
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")


def train_model() -> dict:
    """Train XGBoost model with hyperparameter tuning."""
    X_train, y_train = _load_training_data()

    tuning_results, random_search = _perform_hyperparameter_tuning(X_train, y_train)

    _save_tuning_results(tuning_results)

    best_params = tuning_results["best_params"]
    final_model = _train_final_model(X_train, y_train, best_params)

    cv_results = _evaluate_with_cv(final_model, X_train, y_train)

    _save_model(final_model)

    logging.info("Model training completed successfully")

    return {
        "tuning_results": tuning_results,
        "cv_results": cv_results,
    }


if __name__ == "__main__":
    train_model()
