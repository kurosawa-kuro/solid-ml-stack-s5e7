#!/usr/bin/env python3
"""
Enhanced Gold Minimal Improvement
Based on proven LB 0.974898 → Target: 0.976518 (+0.0016)
Minimal changes to avoid overfitting
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Tuple

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")

DB_PATH = "/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb"

def ensure_medallion_data():
    """Ensure Bronze, Silver, and Gold tables exist"""
    try:
        from src.data.bronze import create_bronze_tables
        from src.data.silver import create_silver_tables
        from src.data.gold import create_gold_tables
        
        create_bronze_tables()
        create_silver_tables()
        create_gold_tables()
        return True
    except Exception as e:
        print(f"Warning: Could not create medallion data: {e}")
        return False

def load_gold_data_safe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Gold layer data"""
    conn = duckdb.connect(DB_PATH)
    train_df = conn.execute("SELECT * FROM gold.train").df()
    test_df = conn.execute("SELECT * FROM gold.test").df()
    conn.close()
    return train_df, test_df

def clean_data_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal data cleaning - same as Enhanced Gold"""
    df = df.copy()
    exclude_cols = ['id', 'Personality', 'Personality_encoded', 'Personality_encoded_1']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Same cleaning as Enhanced Gold
    for col in numeric_cols:
        if df[col].isnull().any():
            if 'ratio' in col.lower() or 'per' in col.lower():
                df[col] = df[col].fillna(0.0)
            elif 'score' in col.lower():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df[col] = df[col].clip(-1e8, 1e8)
    
    return df

def get_enhanced_gold_features(df: pd.DataFrame) -> list:
    """Exact same feature set as Enhanced Gold (LB 0.974898)"""
    exclude_cols = ['id', 'Personality', 'Personality_encoded', 'Personality_encoded_1']
    
    all_features = [col for col in df.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Same priority features as Enhanced Gold
    priority_features = [
        # Original features
        'Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
        'Friends_circle_size', 'Post_frequency',
        
        # Encoded categorical
        'Stage_fear_encoded', 'Drained_after_socializing_encoded',
        
        # Missing indicators
        'Stage_fear_missing', 'Going_outside_missing', 'Time_spent_Alone_missing',
        
        # Basic engineered features
        'social_ratio', 'activity_sum', 'post_per_friend',
        'extrovert_score', 'introvert_score',
        
        # Advanced engineered features
        'Social_event_participation_rate', 'Non_social_outings',
        'Communication_ratio', 'Drain_adjusted_activity',
        'Friend_social_efficiency', 'Activity_ratio',
        'Introvert_extrovert_spectrum', 'Communication_balance',
        
        # Polynomial features
        'poly_extrovert_score_Post_frequency', 'poly_extrovert_score_Social_event_attendance',
        'poly_social_ratio_Post_frequency', 'poly_activity_sum_extrovert_score'
    ]
    
    # Select features that exist
    selected_features = [f for f in priority_features if f in all_features]
    
    # Add remaining up to 50 (Enhanced Gold limit)
    remaining_features = [f for f in all_features if f not in selected_features]
    max_additional = max(0, min(50, 50 - len(selected_features)))
    selected_features.extend(remaining_features[:max_additional])
    
    return selected_features[:50]  # Exactly 50 like Enhanced Gold

def train_enhanced_gold_model(X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
    """Exact same model as Enhanced Gold but with tiny improvements"""
    
    # Enhanced Gold parameters with minimal tweaks for +0.0016
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        num_leaves=52,  # 50 → 52 (minimal increase)
        learning_rate=0.075,  # 0.08 → 0.075 (slightly lower for stability)
        max_depth=8,  # Keep same
        feature_fraction=0.82,  # 0.8 → 0.82 (minimal increase)
        bagging_fraction=0.85,  # Keep same
        bagging_freq=5,  # Keep same
        min_child_samples=20,  # Keep same
        reg_alpha=0.12,  # 0.1 → 0.12 (slightly more regularization)
        reg_lambda=0.1,  # Keep same
        random_state=42,
        n_estimators=220,  # 200 → 220 (minimal increase)
        verbosity=-1,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def cross_validate_minimal(X: np.ndarray, y: np.ndarray) -> dict:
    """Same CV as Enhanced Gold"""
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    feature_importances = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = train_enhanced_gold_model(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)
        scores.append(score)
        
        if hasattr(model, 'feature_importances_'):
            feature_importances.append(model.feature_importances_)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores,
        'feature_importances': np.mean(feature_importances, axis=0) if feature_importances else None
    }

def main():
    """Enhanced Gold Minimal Improvement Pipeline"""
    print("=" * 70)
    print("Enhanced Gold Minimal Improvement - Target: LB +0.0016")
    print("=" * 70)
    print("Base: Enhanced Gold (LB 0.974898)")
    print("Target: Bronze Medal (LB 0.976518)")
    print("Strategy: Minimal parameter tweaks")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. Setup data (same as Enhanced Gold)
    print("\n1. Setting up Enhanced Gold data pipeline...")
    ensure_medallion_data()
    train_df, test_df = load_gold_data_safe()
    print(f"   ✓ Data loaded: train={train_df.shape}, test={test_df.shape}")
    
    # 2. Same cleaning as Enhanced Gold
    print("\n2. Enhanced Gold data cleaning...")
    train_df = clean_data_minimal(train_df)
    test_df = clean_data_minimal(test_df)
    print("   ✓ Data cleaned (Enhanced Gold method)")
    
    # 3. Same features as Enhanced Gold
    print("\n3. Enhanced Gold feature selection...")
    features = get_enhanced_gold_features(train_df)
    X_train = train_df[features].values
    y_train = train_df['Personality_encoded'].values
    X_test = test_df[features].values
    print(f"   ✓ Selected {len(features)} features (Enhanced Gold set)")
    
    # 4. Minimal parameter improvement
    print("\n4. Training with minimal improvements...")
    cv_results = cross_validate_minimal(X_train, y_train)
    
    print(f"   ✓ CV Score: {cv_results['mean_score']:.6f} ± {cv_results['std_score']:.6f}")
    
    # 5. Train final model and generate submission
    print("\n5. Generating minimal improvement submission...")
    final_model = train_enhanced_gold_model(X_train, y_train)
    test_predictions = final_model.predict(X_test)
    
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Extrovert' if pred == 1 else 'Introvert' for pred in test_predictions]
    })
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_gold_minimal_{timestamp}.csv"
    submission_df.to_csv(filename, index=False)
    
    # Final results
    total_time = time.time() - start_time
    bronze_target = 0.976518
    current_lb = 0.974898  # Enhanced Gold LB
    gap_to_bronze = bronze_target - current_lb
    expected_improvement = 0.0016
    
    print("\n" + "=" * 70)
    print("🎯 Enhanced Gold Minimal Improvement Complete")
    print("=" * 70)
    print(f"⏱️  Total Time: {total_time:.1f} seconds")
    print(f"📊 CV Score: {cv_results['mean_score']:.6f} ± {cv_results['std_score']:.6f}")
    print(f"🥇 Base LB: {current_lb:.6f} (Enhanced Gold)")
    print(f"🎯 Target LB: {bronze_target:.6f} (Bronze Medal)")
    print(f"📈 Needed: +{gap_to_bronze:.6f}")
    print(f"🔧 Expected: +{expected_improvement:.6f} (minimal tweaks)")
    print(f"📁 Submission: {filename}")
    print(f"🔧 Features: {len(features)} (same as Enhanced Gold)")
    print(f"⚙️  Changes: num_leaves 50→52, lr 0.08→0.075, n_est 200→220")
    print("=" * 70)
    print("🤞 Expectation: LB 0.974898 → 0.975500+ (Conservative estimate)")
    print("=" * 70)

if __name__ == "__main__":
    main()
