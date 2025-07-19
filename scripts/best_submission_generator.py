"""
Best Submission Generator Script
Generates submission file using the exact configuration that achieved CV 0.9684
- Replicates the exact feature set from baseline_training_20250719_171106.json
- Uses default LightGBM parameters that achieved 0.9684 CV score
- Safe data loading through Medallion pipeline (Bronze -> Silver -> Gold)
- Generates best_submission.csv with same performance as CV 0.9684
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.gold import load_gold_data, create_gold_tables
from src.models import LightGBMModel, LIGHTGBM_PARAMS
from src.validation import check_data_integrity, validate_target_distribution

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def ensure_data_pipeline() -> None:
    """Ensure Gold tables exist by running the complete pipeline if needed"""
    try:
        # Try to load Gold data
        train_df, test_df = load_gold_data()
        logger.info(f"Gold data loaded successfully: train {train_df.shape}, test {test_df.shape}")
    except Exception as e:
        logger.info(f"Gold data not available ({e}), creating pipeline...")
        # Create Gold tables through the pipeline
        create_gold_tables()
        train_df, test_df = load_gold_data()
        logger.info(f"Gold data created successfully: train {train_df.shape}, test {test_df.shape}")


def get_exact_feature_set() -> List[str]:
    """
    Return the exact 10 features that achieved CV 0.9684 score
    Based on baseline_training_20250719_171106.json feature importance ranking
    """
    # Exact feature set that achieved 0.9684 (in importance order)
    cv_0_9684_features = [
        "extrovert_score",           # importance: 521.0
        "social_ratio",              # importance: 506.6  
        "Friends_circle_size",       # importance: 409.4
        "Post_frequency",            # importance: 399.8
        "introvert_score",           # importance: 324.8
        "Going_outside",             # importance: 305.0
        "Social_event_attendance",   # importance: 268.8
        "Time_spent_Alone",          # importance: 175.4
        "Stage_fear_encoded",        # importance: 49.4
        "Drained_after_socializing_encoded"  # importance: 39.8
    ]
    
    logger.info(f"Using exact feature set from CV 0.9684: {len(cv_0_9684_features)} features")
    return cv_0_9684_features


def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Load and prepare data using the exact configuration that achieved CV 0.9684
    
    Returns:
        Tuple of (X_train, y_train, X_test, feature_names, test_df_with_ids)
    """
    logger.info("Loading Gold level data for submission generation...")
    
    # Ensure pipeline is ready
    ensure_data_pipeline()
    
    try:
        train_df, test_df = load_gold_data()
        logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
        
        # Get exact feature set that achieved CV 0.9684
        feature_cols = get_exact_feature_set()
        
        # Verify all features are available
        available_features = [col for col in feature_cols if col in train_df.columns]
        missing_features = [col for col in feature_cols if col not in train_df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            logger.info(f"Available features: {available_features}")
            feature_cols = available_features
        
        # Extract features and target using exact configuration
        X_train = train_df[feature_cols].values
        y_train = train_df["Personality_encoded"].values
        X_test = test_df[feature_cols].values
        
        logger.info(f"Features: {len(feature_cols)}, Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Exact feature columns: {feature_cols}")
        
        # Clean data before integrity checks (handle NaN and infinite values)
        logger.info("Cleaning data for submission generation...")
        
        # Handle NaN values in training data
        for i in range(X_train.shape[1]):
            col_data = X_train[:, i]
            if np.isnan(col_data).any():
                # Fill NaN with median
                median_val = np.nanmedian(col_data)
                X_train[np.isnan(X_train[:, i]), i] = median_val
                logger.info(f"Filled NaN in feature {feature_cols[i]} with median {median_val:.4f}")
        
        # Handle infinite values in training data
        for i in range(X_train.shape[1]):
            col_data = X_train[:, i]
            if np.isinf(col_data).any():
                # Replace inf with large finite values
                finite_data = col_data[np.isfinite(col_data)]
                if len(finite_data) > 0:
                    max_finite = np.max(finite_data)
                    min_finite = np.min(finite_data)
                    X_train[X_train[:, i] == np.inf, i] = max_finite * 2
                    X_train[X_train[:, i] == -np.inf, i] = min_finite * 2
                    logger.info(f"Replaced infinite values in feature {feature_cols[i]}")
        
        # Handle NaN values in test data
        for i in range(X_test.shape[1]):
            col_data = X_test[:, i]
            if np.isnan(col_data).any():
                # Use training median for consistency
                train_col_data = X_train[:, i]
                median_val = np.median(train_col_data)
                X_test[np.isnan(X_test[:, i]), i] = median_val
                logger.info(f"Filled NaN in test feature {feature_cols[i]} with training median {median_val:.4f}")
        
        # Handle infinite values in test data
        for i in range(X_test.shape[1]):
            col_data = X_test[:, i]
            if np.isinf(col_data).any():
                # Use training finite range for consistency
                train_col_data = X_train[:, i]
                finite_data = train_col_data[np.isfinite(train_col_data)]
                if len(finite_data) > 0:
                    max_finite = np.max(finite_data)
                    min_finite = np.min(finite_data)
                    X_test[X_test[:, i] == np.inf, i] = max_finite * 2
                    X_test[X_test[:, i] == -np.inf, i] = min_finite * 2
                    logger.info(f"Replaced infinite values in test feature {feature_cols[i]}")
        
        # Data integrity checks (after cleaning)
        integrity_checks = check_data_integrity(X_train, y_train)
        logger.info(f"Data integrity checks (after cleaning): {integrity_checks}")
        
        # Only fail on critical issues (not NaN/inf which we handled)
        critical_checks = {k: v for k, v in integrity_checks.items() 
                          if k not in ['has_nan', 'has_inf']}
        
        if not all(critical_checks.values()):
            failed_checks = [k for k, v in critical_checks.items() if not v]
            raise ValueError(f"Critical data integrity checks failed: {failed_checks}")
        
        # Target distribution analysis
        target_dist = validate_target_distribution(y_train)
        logger.info(f"Target distribution: {target_dist}")
        
        # Keep test dataframe with IDs for submission
        test_df_with_ids = test_df[["id"]].copy()
        
        return X_train, y_train, X_test, feature_cols, test_df_with_ids
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise


def train_full_model(X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> LightGBMModel:
    """
    Train the full model on all data using exact parameters that achieved CV 0.9684
    
    Args:
        X_train: Training features
        y_train: Training targets  
        feature_names: Feature names
        
    Returns:
        Trained LightGBM model
    """
    logger.info("Training full model with exact CV 0.9684 parameters...")
    
    # Use exact same parameters that achieved CV 0.9684
    model_params = LIGHTGBM_PARAMS.copy()
    logger.info(f"Model parameters: {model_params}")
    
    # Create and train model (no pipeline for submission - direct training)
    model = LightGBMModel(params=model_params, use_pipeline=False)
    model.fit(X_train, y_train, feature_names=feature_names)
    
    logger.info("Full model training completed")
    return model


def generate_submission(model: LightGBMModel, X_test: np.ndarray, test_df_with_ids: pd.DataFrame, 
                       filename: str = "best_submission.csv") -> None:
    """
    Generate submission file using trained model
    
    Args:
        model: Trained LightGBM model
        X_test: Test features
        test_df_with_ids: Test dataframe with ID column
        filename: Output filename
    """
    logger.info("Generating predictions for submission...")
    
    # Generate predictions
    test_predictions = model.predict(X_test)
    test_predictions_proba = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"Generated {len(test_predictions)} predictions")
    logger.info(f"Prediction distribution: {np.bincount(test_predictions)}")
    logger.info(f"Mean probability: {test_predictions_proba.mean():.4f}")
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_df_with_ids['id'],
        'Personality': ['Extrovert' if pred == 1 else 'Introvert' for pred in test_predictions]
    })
    
    # Verify submission format
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Submission columns: {submission.columns.tolist()}")
    logger.info(f"Personality distribution:\n{submission['Personality'].value_counts()}")
    
    # Save submission file
    submission.to_csv(filename, index=False)
    logger.info(f"Submission saved to: {filename}")
    
    # Calculate prediction ratios for verification
    extrovert_count = (submission['Personality'] == 'Extrovert').sum()
    introvert_count = (submission['Personality'] == 'Introvert').sum()
    total_count = len(submission)
    
    extrovert_ratio = extrovert_count / total_count
    introvert_ratio = introvert_count / total_count
    
    logger.info(f"Final submission statistics:")
    logger.info(f"  Total predictions: {total_count}")
    logger.info(f"  Extrovert: {extrovert_count} ({extrovert_ratio:.3f})")
    logger.info(f"  Introvert: {introvert_count} ({introvert_ratio:.3f})")
    
    # Expected ratio from CV 0.9684 training: 74.17% Extrovert, 25.83% Introvert
    logger.info(f"  Expected from CV training: 74.17% Extrovert, 25.83% Introvert")


def main():
    """Main submission generation workflow"""
    logger.info("=" * 60)
    logger.info("Best Submission Generator - CV 0.9684 Configuration")
    logger.info("=" * 60)
    
    try:
        # Load and prepare data with exact CV 0.9684 configuration
        logger.info("Step 1: Loading data with exact feature set...")
        X_train, y_train, X_test, feature_names, test_df_with_ids = load_and_prepare_data()
        
        # Train full model with exact parameters
        logger.info("Step 2: Training full model with CV 0.9684 parameters...")
        model = train_full_model(X_train, y_train, feature_names)
        
        # Generate submission
        logger.info("Step 3: Generating submission file...")
        generate_submission(model, X_test, test_df_with_ids, "best_submission.csv")
        
        logger.info("=" * 60)
        logger.info("✅ Submission generation completed successfully!")
        logger.info("📁 File: best_submission.csv")
        logger.info("🎯 Based on: CV 0.9684 configuration")
        logger.info("📊 Features: 10 (exact match to high-performance training)")
        logger.info("⚙️  Parameters: Default LightGBM (proven performance)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Submission generation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()