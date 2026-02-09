"""Main pipeline orchestrator for Wine classification."""

import logging
import sys
from pathlib import Path

from eda import perform_eda
from evaluate_model import evaluate_model
from feature_engineering import engineer_features
from generate_report import generate_report
from train_model import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)


def _verify_outputs() -> dict[str, bool]:
    """Verify that all expected outputs were created."""
    expected_files = {
        "EDA plots": [
            "output/eda/distributions.png",
            "output/eda/correlation_heatmap.png",
            "output/eda/class_balance.png",
            "output/eda/outliers_summary.png",
        ],
        "Processed data": [
            "output/processed/X_train.csv",
            "output/processed/X_test.csv",
            "output/processed/y_train.csv",
            "output/processed/y_test.csv",
        ],
        "Model artifacts": [
            "output/models/xgboost_model.joblib",
            "output/tuning_results.json",
        ],
        "Evaluation outputs": [
            "output/evaluation/metrics.json",
            "output/evaluation/confusion_matrix.png",
            "output/evaluation/feature_importance.png",
        ],
        "Final report": ["output/report.md"],
    }

    verification_results = {}
    for category, files in expected_files.items():
        all_exist = all(Path(f).exists() for f in files)
        verification_results[category] = all_exist
        if all_exist:
            logging.info(f"✓ {category}: All files created successfully")
        else:
            logging.warning(f"✗ {category}: Some files are missing")
            for f in files:
                if not Path(f).exists():
                    logging.warning(f"  Missing: {f}")

    return verification_results


def run_pipeline() -> None:
    """Run the complete Wine classification pipeline."""
    logging.info("=" * 80)
    logging.info("Starting Wine Classification Pipeline")
    logging.info("=" * 80)

    try:
        logging.info("\n[Step 1/5] Performing Exploratory Data Analysis...")
        perform_eda()
        logging.info("✓ EDA completed")

        logging.info("\n[Step 2/5] Engineering Features...")
        engineer_features()
        logging.info("✓ Feature engineering completed")

        logging.info("\n[Step 3/5] Training Model with Hyperparameter Tuning...")
        train_model()
        logging.info("✓ Model training completed")

        logging.info("\n[Step 4/5] Evaluating Model...")
        evaluate_model()
        logging.info("✓ Model evaluation completed")

        logging.info("\n[Step 5/5] Generating Report...")
        generate_report()
        logging.info("✓ Report generation completed")

        logging.info("\n" + "=" * 80)
        logging.info("Pipeline Execution Summary")
        logging.info("=" * 80)

        verification_results = _verify_outputs()

        all_successful = all(verification_results.values())
        if all_successful:
            logging.info("\n✓ Pipeline completed successfully!")
            logging.info("✓ All output files have been generated")
            logging.info("\nOutput locations:")
            logging.info("  - EDA plots: output/eda/")
            logging.info("  - Processed data: output/processed/")
            logging.info("  - Trained model: output/models/xgboost_model.joblib")
            logging.info("  - Tuning results: output/tuning_results.json")
            logging.info("  - Evaluation metrics: output/evaluation/")
            logging.info("  - Final report: output/report.md")
        else:
            logging.error("\n✗ Pipeline completed with some missing outputs")
            logging.error("Please check the logs above for details")
            sys.exit(1)

    except Exception as e:
        logging.error(f"\n✗ Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
