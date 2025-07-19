#!/usr/bin/env python
"""
Final Bronze Medal Push
Combined best practices: Feature Selection + Optimized Parameters + Larger Sample
Target: 0.976518
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.bronze import load_data
from src.data.silver_enhanced import apply_enhanced_silver_features
from src.models import LightGBMModel

warnings.filterwarnings("ignore")


def get_bronze_medal_params():
    """Best parameters combination for Bronze Medal"""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 250,        # Increased for better performance
        "num_leaves": 70,           # Optimized complexity
        "learning_rate": 0.05,      # Careful learning
        "feature_fraction": 0.85,   # Good feature sampling
        "bagging_fraction": 0.85,   # Good row sampling
        "bagging_freq": 5,
        "min_child_samples": 12,    # Allow finer patterns
        "lambda_l1": 0.02,          # Light regularization
        "lambda_l2": 0.02,
        "min_gain_to_split": 0.01,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "force_row_wise": True,
    }


def select_top_features_lgb(X, y, n_features=120):
    """Select top features using LightGBM importance"""
    print(f"   Selecting top {n_features} features using LightGBM importance...")
    
    # Quick training for feature importance
    model = LightGBMModel(params={
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 50,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "verbosity": -1,
        "random_state": 42,
    })
    model.fit(X, y)
    
    # Get feature importance
    importances = model.model.feature_importances_
    top_indices = np.argsort(importances)[-n_features:]
    selected_features = X.columns[top_indices]
    
    X_selected = X[selected_features]
    
    # Show top 10 features
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"   Top 10 features:")
    for i, (feat, imp) in enumerate(feature_importance_df.head(10).values):
        print(f"     {i+1}. {feat}: {imp:.1f}")
    
    return X_selected, selected_features


def bronze_medal_cv(X, y, n_features=120, folds=5):
    """Final Bronze Medal attempt with all optimizations"""
    print(f"\n🏆 Bronze Medal CV - Final Push")
    print(f"Features: {n_features}, Folds: {folds}")
    
    # Feature selection
    X_selected, selected_features = select_top_features_lgb(X, y, n_features=n_features)
    
    # Best parameters
    params = get_bronze_medal_params()
    
    # Cross validation with more folds for reliability
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_scores = []
    
    print(f"\n   Running {folds}-fold CV...")
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
        fold_start = time.time()
        
        X_train_fold = X_selected.iloc[train_idx]
        X_val_fold = X_selected.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        model = LightGBMModel(params=params)
        model.fit(X_train_fold, y_train_fold)
        
        # Predict and score
        y_pred = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)
        cv_scores.append(score)
        
        fold_time = time.time() - fold_start
        print(f"     Fold {fold+1}: {score:.6f} ({fold_time:.1f}s)")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    elapsed = time.time() - start_time
    gap_to_bronze = 0.976518 - mean_score
    
    print(f"\n   📊 Final Result: {mean_score:.6f} ± {std_score:.6f} ({elapsed:.1f}s)")
    print(f"   🎯 Bronze gap: {gap_to_bronze:+.6f}")
    
    return mean_score, std_score, gap_to_bronze, selected_features


def main():
    """Main Bronze Medal push"""
    print("=" * 80)
    print("🏆 BRONZE MEDAL FINAL PUSH")
    print("=" * 80)
    print("Target: 0.976518")
    print("Current best: ~0.967375")
    print("Strategy: Feature Selection + Optimized Params + Large Sample")
    
    # Load full data for maximum reliability
    print("\n📊 Loading full dataset...")
    train_data, _ = load_data()
    
    # Use large sample for reliable results
    sample_size = min(12000, len(train_data))  # Large sample
    train_sample = train_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    X = train_sample.drop(['id', 'Personality'], axis=1, errors='ignore')
    y = (train_sample['Personality'] == 'Extrovert').astype(int)
    
    print(f"Sample size: {len(train_sample)} rows ({len(train_sample)/len(train_data)*100:.1f}% of data)")
    
    # Apply enhanced features
    print("\n🔧 Applying enhanced Silver features...")
    start_time = time.time()
    X_enhanced = apply_enhanced_silver_features(X, y, is_train=True)
    feature_time = time.time() - start_time
    print(f"   Features: {X.shape[1]} → {X_enhanced.shape[1]} (+{X_enhanced.shape[1] - X.shape[1]})")
    print(f"   Feature engineering: {feature_time:.2f}s")
    
    # Test multiple feature counts
    feature_counts = [100, 120, 150]
    results = []
    
    for n_features in feature_counts:
        print(f"\n{'='*60}")
        print(f"Testing {n_features} features")
        print(f"{'='*60}")
        
        score, std, gap, features = bronze_medal_cv(
            X_enhanced, y, 
            n_features=n_features, 
            folds=5
        )
        
        results.append({
            "n_features": n_features,
            "score": score,
            "std": std,
            "gap": gap,
            "features": features
        })
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏆 BRONZE MEDAL FINAL RESULTS")
    print("=" * 80)
    
    # Sort by performance
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"{'Rank':<4} {'Features':<8} {'Score':<12} {'±Std':<8} {'Gap to Bronze':<12}")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        gap_str = f"{result['gap']:+.6f}"
        print(f"{i:<4} {result['n_features']:<8} {result['score']:.6f} {result['std']:.4f} {gap_str:<12}")
    
    # Best result
    best = results[0]
    print(f"\n🏆 BEST RESULT: {best['n_features']} features")
    print(f"   Score: {best['score']:.6f} ± {best['std']:.6f}")
    print(f"   Bronze gap: {best['gap']:+.6f}")
    
    if best['gap'] <= 0.000:
        print("   🎯 BRONZE MEDAL ACHIEVED! 🏆🏆🏆")
    elif best['gap'] <= 0.002:
        print("   🔥 EXTREMELY CLOSE TO BRONZE! ALMOST THERE!")
    elif best['gap'] <= 0.005:
        print("   ✅ VERY CLOSE TO BRONZE! Final tuning needed!")
    elif best['gap'] <= 0.010:
        print("   📈 SOLID PROGRESS! Getting close to Bronze!")
    else:
        print("   ⚠️  MORE OPTIMIZATION NEEDED")
    
    print(f"\n🚀 Recommended next steps:")
    if best['gap'] <= 0.005:
        print("   - Try ensemble of top models")
        print("   - Fine-tune threshold (may give +0.001-0.002)")
        print("   - Test with full dataset")
    else:
        print("   - More advanced feature engineering")
        print("   - Optuna hyperparameter optimization")
        print("   - Different model architectures")
    
    return results


if __name__ == "__main__":
    results = main()