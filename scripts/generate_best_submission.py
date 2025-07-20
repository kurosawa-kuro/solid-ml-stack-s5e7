#!/usr/bin/env python3
"""
Generate submission file using the best trained model
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from src.data.gold import create_gold_tables
import duckdb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_best_model():
    """Load the most recent heavy model"""
    model_dir = Path("outputs/models")
    heavy_models = list(model_dir.glob("heavy_enhanced_lightgbm_*.pkl"))
    
    if not heavy_models:
        raise FileNotFoundError("No heavy models found")
    
    # Get the most recent model
    latest_model = max(heavy_models, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading model: {latest_model}")
    
    model_data = joblib.load(latest_model)
    
    return model_data


def generate_submission():
    """Generate submission file using best model"""
    logger.info("Starting submission generation...")
    
    # Load model
    model_data = load_best_model()
    models = model_data['cv_results']['models']
    logger.info(f"Loaded {len(models)} CV models")
    logger.info(f"Model CV Score: {model_data['cv_score']:.6f} ± {model_data['cv_std']:.6f}")
    
    # Load test data
    logger.info("Loading test data...")
    # First ensure gold tables exist
    create_gold_tables()
    
    # Connect to DuckDB and load test data
    conn = duckdb.connect("data/kaggle_datasets.duckdb")
    test_df = conn.execute("SELECT * FROM gold.test").fetchdf()
    conn.close()
    
    # Extract features (exclude id column)
    feature_cols = [col for col in test_df.columns if col != 'id']
    X_test = test_df[feature_cols].values
    test_ids = test_df['id'].values
    
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Clean test data (same as training)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Make predictions (ensemble of CV models)
    logger.info("Making predictions...")
    predictions_proba = np.zeros((len(X_test), 2))
    
    for i, model in enumerate(models):
        preds = model.predict_proba(X_test)
        predictions_proba += preds
        logger.info(f"Model {i+1}/{len(models)} predictions complete")
    
    # Average predictions
    predictions_proba /= len(models)
    predictions = (predictions_proba[:, 1] > 0.5).astype(int)
    
    # Map back to original labels
    label_mapping = {0: 'Introvert', 1: 'Extrovert'}
    predictions_labels = [label_mapping[pred] for pred in predictions]
    
    # Create submission
    submission_df = pd.DataFrame({
        'id': test_ids,
        'Personality': predictions_labels
    })
    
    # Save submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f"outputs/submissions/heavy_submission_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    
    logger.info(f"Submission saved to: {submission_path}")
    logger.info(f"Submission shape: {submission_df.shape}")
    logger.info(f"Prediction distribution:")
    logger.info(submission_df['Personality'].value_counts())
    
    # Also save a copy as latest
    latest_path = "outputs/submissions/latest_submission.csv"
    submission_df.to_csv(latest_path, index=False)
    logger.info(f"Also saved as: {latest_path}")
    
    return submission_path


def main():
    """Main function"""
    try:
        submission_path = generate_submission()
        logger.info("=" * 70)
        logger.info("SUBMISSION GENERATION COMPLETE!")
        logger.info(f"File: {submission_path}")
        logger.info("Ready to upload to Kaggle!")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Failed to generate submission: {str(e)}")
        raise


if __name__ == "__main__":
    main()