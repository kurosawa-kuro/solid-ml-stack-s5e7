#!/usr/bin/env python
"""
Feature selection boost for immediate performance improvement
Use feature importance to select the most predictive features
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.bronze import load_data
from src.data.silver_enhanced import apply_enhanced_silver_features
from src.models import LightGBMModel

warnings.filterwarnings("ignore")


def select_features_by_importance(X, y, n_features=100, method="lightgbm"):
    """Select top features using different methods"""
    print(f"   Selecting top {n_features} features using {method}...")
    
    if method == "lightgbm":
        # Train a quick LightGBM to get feature importance
        model = LightGBMModel(params={
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": 50,  # Quick training
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
        
    elif method == "f_classif":
        # Statistical F-test
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
        
    elif method == "mutual_info":
        # Mutual information
        selector = SelectKBest(mutual_info_classif, k=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
    
    X_selected = X[selected_features]
    print(f"   Selected features: {list(selected_features[:5])}...")
    
    return X_selected, selected_features


def test_feature_selection_cv(X, y, n_features_list=[50, 100, 150, 200], method="lightgbm"):
    """Test different numbers of features"""
    print(f"\n🎯 Testing feature selection with {method}")
    
    # Base parameters
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 100,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }
    
    results = []
    
    for n_features in n_features_list:
        print(f"\n🔧 Testing {n_features} features...")
        
        # Select features
        X_selected, selected_features = select_features_by_importance(
            X, y, n_features=n_features, method=method
        )
        
        # Cross validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        
        start_time = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
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
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        elapsed = time.time() - start_time
        gap_to_bronze = 0.976518 - mean_score
        
        print(f"   📊 Result: {mean_score:.6f} ± {std_score:.6f} ({elapsed:.1f}s)")
        print(f"   🎯 Bronze gap: {gap_to_bronze:+.6f}")
        
        results.append({
            "n_features": n_features,
            "method": method,
            "score": mean_score,
            "std": std_score,
            "gap": gap_to_bronze,
            "time": elapsed,
            "selected_features": selected_features
        })
    
    return results


def test_combined_approach(X, y):
    """Test combination of feature selection + parameter tuning"""
    print(f"\n🚀 Testing Combined Approach (Feature Selection + Optimized Params)")
    
    # First, select best features
    X_selected, selected_features = select_features_by_importance(
        X, y, n_features=100, method="lightgbm"
    )
    
    # Optimized parameters for selected features
    optimized_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 200,        # More iterations for selected features
        "num_leaves": 60,           # Higher complexity for quality features
        "learning_rate": 0.06,      # Moderate learning rate
        "feature_fraction": 0.9,    # Use most of the selected features
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_child_samples": 15,
        "lambda_l1": 0.01,          # Light regularization
        "lambda_l2": 0.01,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }
    
    # Cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # More folds for reliability
    cv_scores = []
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
        X_train_fold = X_selected.iloc[train_idx]
        X_val_fold = X_selected.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        model = LightGBMModel(params=optimized_params)
        model.fit(X_train_fold, y_train_fold)
        
        # Predict and score
        y_pred = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)
        cv_scores.append(score)
        
        print(f"   Fold {fold+1}: {score:.6f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    elapsed = time.time() - start_time
    gap_to_bronze = 0.976518 - mean_score
    
    print(f"   📊 Combined Result: {mean_score:.6f} ± {std_score:.6f} ({elapsed:.1f}s)")
    print(f"   🎯 Bronze gap: {gap_to_bronze:+.6f}")
    
    return mean_score, std_score, gap_to_bronze, selected_features


def main():
    """Main function for feature selection boost"""
    print("=" * 80)
    print("🎯 Feature Selection Boost - Bronze Medal Target")
    print("=" * 80)
    print("Target: 0.976518, Current best: ~0.965400")
    print("Strategy: Select most predictive features to reduce noise")
    
    # Load data
    print("\n📊 Loading and preparing data...")
    train_data, _ = load_data()
    
    # Use substantial sample for reliable feature selection
    sample_size = min(8000, len(train_data))
    train_sample = train_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    X = train_sample.drop(['id', 'Personality'], axis=1, errors='ignore')
    y = (train_sample['Personality'] == 'Extrovert').astype(int)
    
    print(f"Sample size: {len(train_sample)} rows")
    
    # Apply enhanced features
    print("🔧 Applying enhanced features...")
    start_time = time.time()
    X_enhanced = apply_enhanced_silver_features(X, y, is_train=True)
    feature_time = time.time() - start_time
    print(f"   Features: {X.shape[1]} → {X_enhanced.shape[1]} (+{X_enhanced.shape[1] - X.shape[1]})")
    print(f"   Feature engineering: {feature_time:.2f}s")
    
    # Test different feature selection methods
    all_results = []
    
    # Method 1: LightGBM importance
    results_lgb = test_feature_selection_cv(
        X_enhanced, y, 
        n_features_list=[50, 100, 150, 200], 
        method="lightgbm"
    )
    all_results.extend(results_lgb)
    
    # Method 2: Statistical F-test
    results_f = test_feature_selection_cv(
        X_enhanced, y,
        n_features_list=[100, 150],  # Test fewer options for speed
        method="f_classif"
    )
    all_results.extend(results_f)
    
    # Method 3: Combined approach
    combined_score, combined_std, combined_gap, selected_features = test_combined_approach(X_enhanced, y)
    all_results.append({
        "n_features": 100,
        "method": "combined",
        "score": combined_score,
        "std": combined_std,
        "gap": combined_gap,
        "selected_features": selected_features
    })
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 FEATURE SELECTION RESULTS")
    print("=" * 80)
    
    # Sort by performance
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"{'Rank':<4} {'Method':<12} {'Features':<8} {'Score':<12} {'Gap':<10}")
    print("-" * 60)
    
    baseline_score = 0.965400
    
    for i, result in enumerate(all_results, 1):
        improvement = result['score'] - baseline_score
        gap_str = f"{result['gap']:+.6f}"
        
        print(f"{i:<4} {result['method']:<12} {result['n_features']:<8} {result['score']:.6f} {gap_str:<10}")
        print(f"     {'Improvement:':<12} {improvement:+.6f}")
    
    # Best result analysis
    best = all_results[0]
    print(f"\n🏆 BEST APPROACH: {best['method']} with {best['n_features']} features")
    print(f"   Score: {best['score']:.6f} ± {best['std']:.6f}")
    print(f"   Improvement: {best['score'] - baseline_score:+.6f} vs baseline")
    print(f"   Bronze gap: {best['gap']:+.6f}")
    
    if best['gap'] <= 0.001:
        print("   🎯 BRONZE MEDAL ACHIEVED! 🏆")
    elif best['gap'] <= 0.003:
        print("   🔥 EXTREMELY CLOSE TO BRONZE!")
    elif best['gap'] <= 0.007:
        print("   ✅ VERY CLOSE TO BRONZE - Final push needed!")
    elif best['gap'] <= 0.015:
        print("   📈 GOOD PROGRESS TO BRONZE")
    else:
        print("   ⚠️  MORE OPTIMIZATION NEEDED")
    
    return all_results


if __name__ == "__main__":
    results = main()