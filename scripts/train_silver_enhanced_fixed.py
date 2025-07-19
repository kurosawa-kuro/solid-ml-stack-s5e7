#!/usr/bin/env python
"""
Fixed Silver Enhanced Training - True 480+ Features Pipeline
Executes: Bronze → Silver Enhanced → Gold → LightGBM
Target: CV 0.976+ (Bronze Medal Achievement)
"""

import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.bronze import load_data, create_bronze_tables
from src.data.silver import create_silver_tables, load_silver_data
from src.data.silver_enhanced import apply_enhanced_silver_features, EnhancedSilverPreprocessor
from src.data.gold import load_gold_data, create_gold_tables
from src.models import LightGBMModel

warnings.filterwarnings("ignore")

def create_silver_enhanced_pipeline():
    """Create complete Silver Enhanced pipeline with 480+ features"""
    print("=" * 80)
    print("Silver Enhanced Pipeline - Bronze Medal Target (CV 0.976+)")
    print("=" * 80)
    
    # Step 1: Ensure all layers exist
    print("\n1. Setting up data layers...")
    try:
        create_bronze_tables()
        print("   ✓ Bronze tables ready")
        
        create_silver_tables() 
        print("   ✓ Silver tables ready")
        
        create_gold_tables()
        print("   ✓ Gold tables ready")
    except Exception as e:
        print(f"   ✗ Layer setup failed: {e}")
        return None, None, None, None
    
    # Step 2: Load processed data
    print("\n2. Loading Gold layer data...")
    try:
        train_df, test_df = load_gold_data()
        print(f"   ✓ Gold data loaded: train={train_df.shape}, test={test_df.shape}")
        
        # Separate features and target  
        id_cols = ["id"]
        target_cols = ["Personality", "Personality_encoded", "Personality_encoded_1"]
        
        # Get feature columns (exclude ID and target)
        feature_cols = [col for col in train_df.columns if col not in id_cols + target_cols]
        
        X_train = train_df[feature_cols]
        # Use the primary encoded target
        y_train = train_df["Personality_encoded"] 
        X_test = test_df[feature_cols]
        
        print(f"   ✓ Features extracted: {len(feature_cols)} features")
        print(f"   ✓ Training samples: {len(X_train)}")
        print(f"   ✓ Test samples: {len(X_test)}")
        
        # Data quality check
        if X_train.isnull().any().any():
            print("   ! NaN values detected, cleaning...")
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
        if np.isinf(X_train.values).any():
            print("   ! Infinite values detected, cleaning...")
            X_train = X_train.replace([np.inf, -np.inf], 0)
            X_test = X_test.replace([np.inf, -np.inf], 0)
            
        print(f"   ✓ Data quality validated")
        
        return X_train, y_train, X_test, feature_cols
        
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        return None, None, None, None

def train_enhanced_model(X_train, y_train, feature_cols):
    """Train LightGBM with enhanced features"""
    print("\n3. Training Enhanced LightGBM Model...")
    
    # Enhanced LightGBM parameters for high feature count
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss', 
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,  # Important for 480+ features
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    
    # 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    feature_importance = pd.DataFrame()
    
    print(f"   Starting 5-fold CV with {len(feature_cols)} features...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"   Fold {fold + 1}/5...", end=" ")
        
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Create datasets
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Validate
        val_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
        val_pred_binary = (val_pred > 0.5).astype(int)
        fold_score = accuracy_score(y_fold_val, val_pred_binary)
        
        cv_scores.append(fold_score)
        models.append(model)
        
        # Feature importance
        fold_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain'),
            'fold': fold
        })
        feature_importance = pd.concat([feature_importance, fold_importance])
        
        print(f"Score: {fold_score:.4f}")
    
    # CV Results
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    print(f"\n   📊 Cross-Validation Results:")
    print(f"      Mean CV Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"      Individual Scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    # Feature importance summary
    importance_summary = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    print(f"\n   🔝 Top 10 Features:")
    for i, (feature, importance) in enumerate(importance_summary.head(10).items()):
        print(f"      {i+1:2d}. {feature}: {importance:.1f}")
    
    # Success evaluation
    bronze_target = 0.976518
    gap_to_bronze = bronze_target - mean_score
    
    print(f"\n   🎯 Bronze Medal Analysis:")
    print(f"      Current Score:  {mean_score:.6f}")
    print(f"      Bronze Target:  {bronze_target:.6f}")
    print(f"      Gap to Close:   {gap_to_bronze:+.6f} ({gap_to_bronze*100:+.3f}%)")
    
    if gap_to_bronze <= 0:
        print(f"      🏆 BRONZE MEDAL ACHIEVED!")
    elif gap_to_bronze <= 0.005:
        print(f"      🔥 Very close to Bronze! Hyperparameter tuning recommended.")
    elif gap_to_bronze <= 0.01:
        print(f"      📈 Good progress! Feature engineering or ensemble needed.")
    else:
        print(f"      ⚠️  Significant gap remains. Pipeline validation needed.")
    
    return models[0], mean_score, std_score, importance_summary

def generate_submission(model, X_test, test_df):
    """Generate submission file"""
    print("\n4. Generating submission...")
    
    # Predict
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    test_pred_binary = (test_pred > 0.5).astype(int)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Extrovert' if pred == 1 else 'Introvert' for pred in test_pred_binary]
    })
    
    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"silver_enhanced_submission_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    
    print(f"   ✓ Submission saved: {filename}")
    print(f"   ✓ Predictions: {len(submission)} rows")
    print(f"   ✓ Distribution: {submission['Personality'].value_counts().to_dict()}")
    
    return filename

def main():
    """Main execution pipeline"""
    start_time = time.time()
    
    # Create enhanced pipeline
    X_train, y_train, X_test, feature_cols = create_silver_enhanced_pipeline()
    
    if X_train is None:
        print("\n❌ Pipeline creation failed. Exiting.")
        return
    
    # Train model
    model, cv_score, cv_std, feature_importance = train_enhanced_model(X_train, y_train, feature_cols)
    
    # Generate submission
    test_df = load_gold_data()[1]
    submission_file = generate_submission(model, X_test, test_df)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n" + "="*80)
    print(f"🏁 Silver Enhanced Training Complete")
    print(f"⏱️  Total Time: {total_time:.1f} seconds")
    print(f"📊 CV Score: {cv_score:.4f} ± {cv_std:.4f}")
    print(f"📁 Submission: {submission_file}")
    print(f"🎯 Bronze Gap: {0.976518 - cv_score:+.6f}")
    print("="*80)

if __name__ == "__main__":
    main()
