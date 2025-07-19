#!/usr/bin/env python3
"""
Bronze Medal Optimized Script
Builds on Enhanced Gold (LB 0.974898) → Target: 0.976518 (+0.0016)
Optimizations: Hyperparameter tuning + Feature selection + Ensemble
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
import optuna

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

def clean_data_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced data cleaning for Bronze medal target"""
    df = df.copy()
    exclude_cols = ['id', 'Personality', 'Personality_encoded', 'Personality_encoded_1']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Advanced missing value handling
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
    
    # Clean infinite and extreme values
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        df[col] = df[col].replace([np.inf, -np.inf], 0.0)
        # Clip extreme values
        df[col] = df[col].clip(-1e6, 1e6)
    
    return df

def get_optimized_features(df: pd.DataFrame, max_features: int = 60) -> list:
    """Get optimized feature set for Bronze medal"""
    exclude_cols = ['id', 'Personality', 'Personality_encoded', 'Personality_encoded_1']
    
    all_features = [col for col in df.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Enhanced priority features for Bronze medal
    priority_features = [
        # Core features
        'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
        'Friends_circle_size', 'Post_frequency',
        'Stage_fear_encoded', 'Drained_after_socializing_encoded',
        
        # Missing indicators
        'Stage_fear_missing', 'Going_outside_missing', 'Time_spent_Alone_missing',
        
        # Basic engineered features
        'social_ratio', 'activity_sum', 'post_per_friend',
        'extrovert_score', 'introvert_score',
        
        # Winner solution features
        'Social_event_participation_rate', 'Non_social_outings',
        'Communication_ratio', 'Drain_adjusted_activity', 
        'Friend_social_efficiency', 'Activity_ratio',
        'Introvert_extrovert_spectrum', 'Communication_balance',
        
        # High-importance polynomial features
        'poly_extrovert_score_Post_frequency', 
        'poly_extrovert_score_Social_event_attendance',
        'poly_social_ratio_Post_frequency', 
        'poly_activity_sum_extrovert_score',
        'poly_extrovert_score_Going_outside',
        'poly_social_ratio_Going_outside',
        
        # Top statistical features
        'personality_balance', 'extrovert_avg', 'personality_ratio',
        'activity_std', 'introvert_avg',
        
        # Interaction features
        'extrovert_social_interaction', 'social_friends_interaction',
        'fear_drain_interaction', 'extrovert_drain_interaction'
    ]
    
    # Select existing features
    selected_features = [f for f in priority_features if f in all_features]
    
    # Add remaining high-importance features
    remaining_features = [f for f in all_features if f not in selected_features]
    max_additional = max(0, min(max_features - len(selected_features), len(remaining_features)))
    selected_features.extend(remaining_features[:max_additional])
    
    return selected_features[:max_features]

def objective(trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective function for Bronze medal hyperparameter optimization"""
    
    # Enhanced parameter space for Bronze medal
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.3),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.3),
        'random_state': 42,
        'verbosity': -1
    }
    
    # 3-Fold CV for speed
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=300,
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        
        val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        val_pred_binary = (val_pred > 0.5).astype(int)
        score = accuracy_score(y_val_fold, val_pred_binary)
        scores.append(score)
    
    return np.mean(scores)

def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 100) -> dict:
    """Optimize hyperparameters for Bronze medal"""
    print(f"\n🔧 Optimizing hyperparameters ({n_trials} trials)...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    print(f"   ✓ Best score: {study.best_value:.6f}")
    print(f"   ✓ Best params: {study.best_params}")
    
    return study.best_params

def train_optimized_ensemble(X_train: np.ndarray, y_train: np.ndarray, 
                           best_params: dict) -> list:
    """Train ensemble of optimized models"""
    print("\n🎯 Training optimized ensemble...")
    
    models = []
    
    # Model 1: Optimized parameters
    model1_params = best_params.copy()
    model1_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': 42,
        'verbosity': -1
    })
    
    # Model 2: Conservative parameters (for diversity)
    model2_params = best_params.copy()
    model2_params.update({
        'learning_rate': best_params['learning_rate'] * 0.8,
        'num_leaves': max(20, best_params['num_leaves'] - 10),
        'random_state': 1337,
        'verbosity': -1
    })
    
    # Model 3: Aggressive parameters (for diversity)
    model3_params = best_params.copy()
    model3_params.update({
        'learning_rate': min(0.15, best_params['learning_rate'] * 1.2),
        'num_leaves': min(80, best_params['num_leaves'] + 10),
        'random_state': 2023,
        'verbosity': -1
    })
    
    param_sets = [model1_params, model2_params, model3_params]
    
    for i, params in enumerate(param_sets):
        print(f"   Training model {i+1}/3...", end=" ")
        
        model = lgb.LGBMClassifier(**params, n_estimators=500)
        model.fit(X_train, y_train)
        models.append(model)
        
        print("✓")
    
    return models

def ensemble_predict(models: list, X_test: np.ndarray) -> np.ndarray:
    """Ensemble prediction with weighted voting"""
    predictions = []
    
    for model in models:
        pred_proba = model.predict_proba(X_test)[:, 1]
        predictions.append(pred_proba)
    
    # Weighted average (first model gets more weight as it's the best)
    weights = [0.5, 0.3, 0.2]
    ensemble_proba = np.average(predictions, axis=0, weights=weights)
    
    return (ensemble_proba > 0.5).astype(int)

def evaluate_ensemble(models: list, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate ensemble with 5-fold CV"""
    print("\n📊 Evaluating ensemble with 5-fold CV...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train ensemble on fold
        fold_models = []
        for model in models:
            fold_model = lgb.LGBMClassifier(**model.get_params())
            fold_model.fit(X_train_fold, y_train_fold)
            fold_models.append(fold_model)
        
        # Ensemble prediction
        predictions = []
        for fold_model in fold_models:
            pred_proba = fold_model.predict_proba(X_val_fold)[:, 1]
            predictions.append(pred_proba)
        
        weights = [0.5, 0.3, 0.2]
        ensemble_proba = np.average(predictions, axis=0, weights=weights)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        score = accuracy_score(y_val_fold, ensemble_pred)
        scores.append(score)
        
        print(f"   Fold {fold+1}: {score:.6f}")
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores
    }

def main():
    """Bronze Medal Optimization Pipeline"""
    print("=" * 80)
    print("Bronze Medal Optimization - Target: 0.976518 (LB +0.0016)")
    print("=" * 80)
    
    start_time = time.time()
    
    # 1. Setup data
    print("\n1. Setting up optimized data pipeline...")
    ensure_medallion_data()
    train_df, test_df = load_gold_data_safe()
    print(f"   ✓ Data loaded: train={train_df.shape}, test={test_df.shape}")
    
    # 2. Advanced data cleaning
    print("\n2. Advanced data cleaning...")
    train_df = clean_data_advanced(train_df)
    test_df = clean_data_advanced(test_df)
    print("   ✓ Data cleaned")
    
    # 3. Optimized feature selection
    print("\n3. Optimized feature selection...")
    features = get_optimized_features(train_df, max_features=60)
    X_train = train_df[features].values
    y_train = train_df['Personality_encoded'].values
    X_test = test_df[features].values
    print(f"   ✓ Selected {len(features)} optimized features")
    
    # 4. Hyperparameter optimization
    best_params = optimize_hyperparameters(X_train, y_train, n_trials=50)
    
    # 5. Train optimized ensemble
    models = train_optimized_ensemble(X_train, y_train, best_params)
    
    # 6. Evaluate ensemble
    cv_results = evaluate_ensemble(models, X_train, y_train)
    
    # 7. Generate submission
    print("\n🎯 Generating Bronze Medal submission...")
    ensemble_predictions = ensemble_predict(models, X_test)
    
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Personality': ['Extrovert' if pred == 1 else 'Introvert' for pred in ensemble_predictions]
    })
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"bronze_medal_submission_{timestamp}.csv"
    submission_df.to_csv(filename, index=False)
    
    # Final results
    total_time = time.time() - start_time
    bronze_target = 0.976518
    gap_to_bronze = bronze_target - cv_results['mean_score']
    
    print("\n" + "=" * 80)
    print("🏆 Bronze Medal Optimization Complete")
    print("=" * 80)
    print(f"⏱️  Total Time: {total_time:.1f} seconds")
    print(f"📊 Ensemble CV: {cv_results['mean_score']:.6f} ± {cv_results['std_score']:.6f}")
    print(f"🎯 Bronze Target: {bronze_target:.6f}")
    print(f"📈 Gap to Bronze: {gap_to_bronze:+.6f}")
    print(f"📁 Submission: {filename}")
    print(f"🔧 Features Used: {len(features)}")
    
    if gap_to_bronze <= 0:
        print("🥉 BRONZE MEDAL ACHIEVED! 🥉")
    elif gap_to_bronze <= 0.003:
        print("🔥 VERY CLOSE TO BRONZE MEDAL! 🔥")
    else:
        print("📈 Good progress towards Bronze Medal")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
