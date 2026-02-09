import logging
from pathlib import Path

import joblib
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from .feature_engineering import process_data
from .utils import setup_logging


def train_model() -> None:
    """Train XGBoost model with RandomizedSearchCV."""
    setup_logging()
    logging.info("Starting Model Training...")

    # Load and process data
    X_train, X_test, y_train, y_test, scaler = process_data()

    # Define model
    xgb = XGBClassifier(
        objective="multi:softmax", eval_metric="mlogloss", use_label_encoder=False, random_state=42
    )

    # Define hyperparameter grid
    param_dist = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    # RandomizedSearchCV
    clf = RandomizedSearchCV(
        xgb, param_distributions=param_dist, n_iter=20, cv=5, verbose=1, random_state=42, n_jobs=-1
    )

    logging.info("Running RandomizedSearchCV...")
    clf.fit(X_train, y_train)

    logging.info(f"Best parameters: {clf.best_params_}")
    logging.info(f"Best cross-validation score: {clf.best_score_:.4f}")

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    joblib.dump(clf.best_estimator_, model_path)
    logging.info(f"Model saved to {model_path}")

    # Also save the scaler for reproducibility if needed (though process_data recreates it)
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    train_model()
