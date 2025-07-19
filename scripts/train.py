"""
ベースラインモデル学習実行
- データ読み込み（Goldレベル）
- モデル学習・CV実行
- 結果出力・保存
- 通知・ログ出力
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.gold import load_gold_data
from src.models import CrossValidationTrainer, LightGBMModel, save_model_with_metadata
from src.util.notifications import notify_complete, notify_error, notify_start
from src.util.time_tracker import WorkflowTimer, WorkflowTimeTracker
from src.validation import CVLogger, check_data_integrity, validate_target_distribution

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Ensure output directory exists for logging
Path("outputs/logs").mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("outputs/logs/training.log", mode="a")],
)
logger = logging.getLogger(__name__)


def create_output_directories():
    """Create necessary output directories"""
    output_dirs = ["outputs/models", "outputs/logs", "outputs/submissions"]

    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {dir_path}")


def initialize_tracking() -> WorkflowTimeTracker:
    """Initialize time tracking"""
    tracker_path = "outputs/logs/workflow_times.json"
    tracker = WorkflowTimeTracker(tracker_path)
    logger.info("Time tracker initialized")
    return tracker


def load_and_prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load and prepare data for training

    Returns:
        Tuple of (X_train, y_train, X_test, feature_names)
    """
    logger.info("Loading Gold level data...")

    try:
        train_df, test_df = load_gold_data()
        logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

        # Separate features and target
        id_cols = ["id"]
        target_cols = ["Personality", "Personality_encoded"]

        # Get feature columns
        feature_cols = [col for col in train_df.columns if col not in id_cols + target_cols]

        # Extract features and target
        X_train = train_df[feature_cols].values
        y_train = train_df["Personality_encoded"].values
        X_test = test_df[feature_cols].values

        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X_train)}")
        logger.info(f"Feature columns: {feature_cols}")

        # Data integrity checks
        integrity_checks = check_data_integrity(X_train, y_train)
        logger.info(f"Data integrity checks: {integrity_checks}")

        if not all(integrity_checks.values()):
            failed_checks = [k for k, v in integrity_checks.items() if not v]
            raise ValueError(f"Data integrity checks failed: {failed_checks}")

        # Target distribution analysis
        target_dist = validate_target_distribution(y_train)
        logger.info(f"Target distribution: {target_dist}")

        return X_train, y_train, X_test, feature_cols

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise


def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]) -> Dict[str, Any]:
    """
    Train baseline LightGBM model with cross-validation

    Args:
        X_train: Training features
        y_train: Training targets
        feature_names: Feature names

    Returns:
        Cross-validation results
    """
    logger.info("Starting LightGBM baseline training...")

    # Initialize CV trainer
    cv_trainer = CrossValidationTrainer()

    # Send training start notification
    model_config = {"model_type": "LightGBM", "features": len(feature_names), "samples": len(X_train), "cv_folds": 5}

    try:
        notify_start("LightGBM Baseline", model_config)
    except Exception as e:
        logger.warning(f"Failed to send start notification: {str(e)}")

    # Perform cross-validation
    try:
        cv_results = cv_trainer.train_cv(model_class=LightGBMModel, X=X_train, y=y_train, feature_names=feature_names)

        logger.info(f"CV Score: {cv_results['mean_score']: .4f} ± {cv_results['std_score']: .4f}")
        logger.info(f"CV AUC: {cv_results['mean_auc']: .4f} ± {cv_results['std_auc']: .4f}")
        logger.info(f"Training time: {cv_results['training_time']: .1f} seconds")

        return cv_results

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        try:
            notify_error("training", str(e))
        except Exception:
            pass
        raise


def save_results(cv_results: Dict[str, Any], feature_names: list[str]) -> None:
    """
    Save training results and models

    Args:
        cv_results: Cross-validation results
        feature_names: Feature names
    """
    logger.info("Saving results...")

    # Save the best model (first fold for simplicity)
    best_model = cv_results["models"][0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model save path
    model_path = f"outputs/models/lightgbm_baseline_{timestamp}.pkl"

    # Additional metadata
    metadata = {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "training_samples": len(cv_results["oof_predictions"]),
        "prediction_distribution": cv_results["prediction_distribution"],
    }

    # Save model with metadata
    save_model_with_metadata(best_model, cv_results, model_path, metadata)

    # Save CV log
    cv_logger = CVLogger()
    log_entry = cv_logger.create_log_entry(
        model_type="LightGBM",
        cv_config=cv_results["cv_config"],
        fold_scores=cv_results["fold_scores"],
        training_time=cv_results["training_time"],
        feature_count=len(feature_names),
        data_samples=len(cv_results["oof_predictions"]),
        prediction_distribution=cv_results["prediction_distribution"],
        feature_importance=(
            cv_results.get("feature_importance", {}).to_dict("records")
            if cv_results.get("feature_importance") is not None
            else None
        ),
    )

    log_path = cv_logger.save_json_log(log_entry, f"baseline_training_{timestamp}.json")

    # Save fold-wise results as CSV
    fold_results_df = pd.DataFrame(
        {
            "fold": range(len(cv_results["fold_scores"])),
            "accuracy": cv_results["fold_scores"],
            "auc": cv_results["fold_auc_scores"],
        }
    )

    csv_path = cv_logger.save_csv_log(fold_results_df, f"fold_results_{timestamp}.csv")

    logger.info(f"Results saved: model={model_path}, log={log_path}, csv={csv_path}")


def send_completion_notification(cv_results: Dict[str, Any]) -> None:
    """
    Send training completion notification

    Args:
        cv_results: Cross-validation results
    """
    try:
        metrics = {
            "accuracy": cv_results["mean_score"],
            "accuracy_std": cv_results["std_score"],
            "auc": cv_results["mean_auc"],
            "auc_std": cv_results["std_auc"],
            "extrovert_ratio": cv_results["prediction_distribution"]["extrovert_ratio"],
        }

        notify_complete(model_name="LightGBM Baseline", metrics=metrics, duration=cv_results["training_time"])

        logger.info("Completion notification sent")

    except Exception as e:
        logger.warning(f"Failed to send completion notification: {str(e)}")


def evaluate_success_criteria(cv_results: Dict[str, Any]) -> Dict[str, bool]:
    """
    Evaluate if training meets success criteria

    Args:
        cv_results: Cross-validation results

    Returns:
        Dictionary of success criteria evaluation
    """
    target_score = 0.975
    stability_threshold = 0.002
    max_training_time = 300  # 5 minutes

    criteria = {
        "score_achieved": cv_results["mean_score"] >= target_score,
        "stable_performance": cv_results["std_score"] <= stability_threshold,
        "reasonable_time": cv_results["training_time"] <= max_training_time,
        "data_integrity": all(cv_results["data_integrity_checks"].values()),
    }

    overall_success = all(criteria.values())
    criteria["overall_success"] = overall_success

    return criteria


def main():
    """Main training workflow"""
    logger.info("=" * 50)
    logger.info("Starting LightGBM Baseline Training")
    logger.info("=" * 50)

    # Initialize
    create_output_directories()
    tracker = initialize_tracking()

    try:
        with WorkflowTimer(tracker, "baseline_training"):
            # Load and prepare data
            with WorkflowTimer(tracker, "data_preparation"):
                X_train, y_train, X_test, feature_names = load_and_prepare_data()

            # Train model
            with WorkflowTimer(tracker, "model_training"):
                cv_results = train_baseline_model(X_train, y_train, feature_names)

            # Save results
            with WorkflowTimer(tracker, "save_results"):
                save_results(cv_results, feature_names)

            # Evaluate success
            success_criteria = evaluate_success_criteria(cv_results)
            logger.info(f"Success criteria: {success_criteria}")

            # Send completion notification
            send_completion_notification(cv_results)

            # Final summary
            logger.info("=" * 50)
            logger.info("Training Summary: ")
            logger.info(f"CV Accuracy: {cv_results['mean_score']: .4f} ± {cv_results['std_score']: .4f}")
            logger.info(f"CV AUC: {cv_results['mean_auc']: .4f} ± {cv_results['std_auc']: .4f}")
            logger.info(f"Training Time: {cv_results['training_time']: .1f} seconds")
            logger.info(f"Target Achievement: {success_criteria['score_achieved']}")
            logger.info(f"Stability: {success_criteria['stable_performance']}")
            logger.info("=" * 50)

            if success_criteria["overall_success"]:
                logger.info("🎉 All success criteria met!")
            else:
                logger.warning("⚠️ Some success criteria not met")

    except Exception as e:
        logger.error(f"Training workflow failed: {str(e)}")
        try:
            notify_error("workflow", str(e))
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
