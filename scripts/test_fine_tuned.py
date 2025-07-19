#!/usr/bin/env python
"""
Test fine-tuned parameters for immediate performance boost
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.fine_tuned_params import FINE_TUNED_PARAMS, get_ensemble_params
from src.data.bronze import load_data
from src.data.silver_enhanced import apply_enhanced_silver_features
from src.models import LightGBMModel


def test_single_params(params, X, y, folds=3, name="Unknown"):
    """Test a single parameter set"""
    print(f"\n🔧 Testing {name}...")
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_scores = []
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        model = LightGBMModel(params=params)
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
    
    print(f"   📊 Result: {mean_score:.6f} ± {std_score:.6f} ({elapsed:.1f}s)")
    print(f"   🎯 Bronze gap: {gap_to_bronze:+.6f}")
    
    return mean_score, std_score, gap_to_bronze


def test_ensemble_approach(X, y, folds=3):
    """Test ensemble of multiple models"""
    print(f"\n🎭 Testing Ensemble Approach...")
    
    ensemble_params = get_ensemble_params()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train ensemble of models
        predictions = []
        for i, params in enumerate(ensemble_params):
            model = LightGBMModel(params=params)
            model.fit(X_train_fold, y_train_fold)
            y_pred_proba = model.model.predict_proba(X_val_fold)[:, 1]
            predictions.append(y_pred_proba)
        
        # Average predictions
        ensemble_proba = np.mean(predictions, axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        score = accuracy_score(y_val_fold, ensemble_pred)
        fold_scores.append(score)
        
        print(f"   Fold {fold+1}: {score:.6f} (ensemble of {len(ensemble_params)} models)")
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    gap_to_bronze = 0.976518 - mean_score
    
    print(f"   📊 Ensemble Result: {mean_score:.6f} ± {std_score:.6f}")
    print(f"   🎯 Bronze gap: {gap_to_bronze:+.6f}")
    
    return mean_score, std_score, gap_to_bronze


def main():
    """Main function to test all fine-tuned parameters"""
    print("=" * 80)
    print("🚀 Fine-Tuned Parameter Testing - Bronze Medal Push")
    print("=" * 80)
    print("Target: 0.976518, Current best: ~0.968689")
    
    # Load data
    print("\n📊 Loading and preparing data...")
    train_data, _ = load_data()
    
    # Use larger sample for better reliability
    sample_size = min(5000, len(train_data))
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
    
    # Test all fine-tuned parameter sets
    results = []
    
    for name, params in FINE_TUNED_PARAMS.items():
        mean_score, std_score, gap = test_single_params(
            params, X_enhanced, y, folds=3, name=name
        )
        results.append({
            "name": name,
            "score": mean_score,
            "std": std_score,
            "gap": gap
        })
    
    # Test ensemble approach
    ensemble_score, ensemble_std, ensemble_gap = test_ensemble_approach(X_enhanced, y, folds=3)
    results.append({
        "name": "ensemble",
        "score": ensemble_score,
        "std": ensemble_std,
        "gap": ensemble_gap
    })
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 FINE-TUNING RESULTS")
    print("=" * 80)
    
    # Sort by performance
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"{'Rank':<4} {'Method':<16} {'Score':<12} {'±Std':<8} {'Gap to Bronze':<12}")
    print("-" * 60)
    
    baseline_score = 0.968689
    
    for i, result in enumerate(results, 1):
        improvement = result['score'] - baseline_score
        gap_str = f"{result['gap']:+.6f}"
        improvement_str = f"{improvement:+.6f}"
        
        print(f"{i:<4} {result['name']:<16} {result['score']:.6f} {result['std']:.4f} {gap_str:<12}")
        print(f"     {'Improvement:':<16} {improvement_str}")
    
    # Best result analysis
    best = results[0]
    print(f"\n🏆 BEST METHOD: {best['name']}")
    print(f"   Score: {best['score']:.6f} ± {best['std']:.6f}")
    print(f"   Improvement: {best['score'] - baseline_score:+.6f} vs baseline")
    print(f"   Bronze gap: {best['gap']:+.6f}")
    
    if best['gap'] <= 0.001:
        print("   🎯 BRONZE MEDAL ACHIEVED! 🏆")
    elif best['gap'] <= 0.003:
        print("   🔥 EXTREMELY CLOSE TO BRONZE!")
    elif best['gap'] <= 0.007:
        print("   ✅ VERY GOOD PROGRESS TO BRONZE")
    elif best['gap'] <= 0.015:
        print("   📈 SOLID PROGRESS TO BRONZE")
    else:
        print("   ⚠️  MORE OPTIMIZATION NEEDED")
    
    return results


if __name__ == "__main__":
    results = main()