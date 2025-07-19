#!/usr/bin/env python
"""
Hyperparameter benchmark for immediate performance improvement
Tests multiple parameter sets and finds the best one
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.quick_cv_enhanced import run_quick_cv_with_enhanced_features
from scripts.optimized_params import PARAM_SETS, get_best_params_for_target
from src.data.bronze import load_data
from src.data.silver_enhanced import apply_enhanced_silver_features
from src.models import LightGBMModel
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


def benchmark_single_params(params, X, y, sample_ratio=0.1, folds=3, name="Unknown"):
    """Benchmark a single parameter set"""
    print(f"\n🔧 Testing {name}...")
    print(f"   n_estimators: {params['n_estimators']}, num_leaves: {params['num_leaves']}")
    print(f"   learning_rate: {params['learning_rate']}, regularization: L1={params.get('lambda_l1', 0)}")
    
    start_time = time.time()
    
    # Sample data for speed
    if sample_ratio < 1.0:
        n_samples = int(len(X) * sample_ratio)
        sample_idx = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[sample_idx].reset_index(drop=True)
        y_sample = y.iloc[sample_idx].reset_index(drop=True)
    else:
        X_sample, y_sample = X, y
    
    # Cross validation
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sample, y_sample)):
        X_train_fold = X_sample.iloc[train_idx]
        X_val_fold = X_sample.iloc[val_idx]
        y_train_fold = y_sample.iloc[train_idx]
        y_val_fold = y_sample.iloc[val_idx]
        
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
    
    print(f"   📊 Result: {mean_score:.6f} ± {std_score:.6f} ({elapsed:.1f}s)")
    print(f"   🎯 Bronze gap: {0.976518 - mean_score:+.6f}")
    
    return {
        "name": name,
        "mean_score": mean_score,
        "std_score": std_score,
        "elapsed": elapsed,
        "gap_to_bronze": 0.976518 - mean_score,
        "params": params
    }


def run_hyperparameter_benchmark(sample_ratio=0.1, folds=3, max_params=None):
    """Run comprehensive hyperparameter benchmark"""
    print("=" * 80)
    print("🚀 Hyperparameter Benchmark - Bronze Medal Target")
    print("=" * 80)
    print(f"Sample: {sample_ratio*100:.1f}%, Folds: {folds}")
    print(f"Current baseline: 0.968689, Target: 0.976518 (+0.007829 needed)")
    
    # Load and prepare data
    print("\n📊 Loading data...")
    train_data, _ = load_data()
    
    if sample_ratio < 1.0:
        n_samples = int(len(train_data) * sample_ratio * 1.5)  # Extra for feature engineering
        train_data = train_data.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    X = train_data.drop(['id', 'Personality'], axis=1, errors='ignore')
    y = (train_data['Personality'] == 'Extrovert').astype(int)
    
    # Apply enhanced features
    print("🔧 Applying enhanced features...")
    start_time = time.time()
    X_enhanced = apply_enhanced_silver_features(X, y, is_train=True)
    feature_time = time.time() - start_time
    print(f"   Features: {X.shape[1]} → {X_enhanced.shape[1]} (+{X_enhanced.shape[1] - X.shape[1]})")
    print(f"   Feature engineering: {feature_time:.2f}s")
    
    # Benchmark all parameter sets
    results = []
    param_names = list(PARAM_SETS.keys())
    
    if max_params:
        param_names = param_names[:max_params]
    
    for name in param_names:
        params = PARAM_SETS[name]
        result = benchmark_single_params(
            params, X_enhanced, y, 
            sample_ratio=sample_ratio, 
            folds=folds, 
            name=name
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 BENCHMARK RESULTS")
    print("=" * 80)
    
    # Sort by performance
    results.sort(key=lambda x: x['mean_score'], reverse=True)
    
    print(f"{'Rank':<4} {'Name':<18} {'Score':<12} {'±Std':<8} {'Gap':<10} {'Time':<6}")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        gap_str = f"{result['gap_to_bronze']:+.6f}"
        print(f"{i:<4} {result['name']:<18} {result['mean_score']:.6f} {result['std_score']:.4f} {gap_str:<10} {result['elapsed']:.1f}s")
    
    # Best result
    best = results[0]
    print(f"\n🏆 BEST PERFORMER: {best['name']}")
    print(f"   Score: {best['mean_score']:.6f} ± {best['std_score']:.6f}")
    print(f"   Improvement: {best['mean_score'] - 0.968689:+.6f} vs baseline")
    print(f"   Bronze gap: {best['gap_to_bronze']:+.6f}")
    
    if best['gap_to_bronze'] <= 0.001:
        print("   🎯 BRONZE MEDAL ACHIEVED!")
    elif best['gap_to_bronze'] <= 0.005:
        print("   🔥 VERY CLOSE TO BRONZE!")
    elif best['gap_to_bronze'] <= 0.010:
        print("   ✅ STRONG PROGRESS TO BRONZE")
    else:
        print("   ⚠️  MORE OPTIMIZATION NEEDED")
    
    return results


def quick_test_best_params(sample_ratio=0.05, folds=2):
    """Quick test of the best parameter set"""
    print("🚀 Quick test of best parameters...")
    
    best_params = get_best_params_for_target("bronze_medal")
    
    # Load data
    train_data, _ = load_data()
    n_samples = int(len(train_data) * sample_ratio)
    train_sample = train_data.sample(n=n_samples, random_state=42)
    
    X = train_sample.drop(['id', 'Personality'], axis=1, errors='ignore')
    y = (train_sample['Personality'] == 'Extrovert').astype(int)
    X_enhanced = apply_enhanced_silver_features(X, y, is_train=True)
    
    # Test
    result = benchmark_single_params(
        best_params, X_enhanced, y,
        sample_ratio=1.0, folds=folds,
        name="competition_tuned"
    )
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter benchmark")
    parser.add_argument("--sample", type=float, default=0.1, help="Sample ratio")
    parser.add_argument("--folds", type=int, default=3, help="CV folds")
    parser.add_argument("--quick", action="store_true", help="Quick test only")
    parser.add_argument("--max-params", type=int, help="Max number of param sets to test")
    
    args = parser.parse_args()
    
    if args.quick:
        result = quick_test_best_params()
        print(f"\nQuick test result: {result['mean_score']:.6f}")
    else:
        results = run_hyperparameter_benchmark(
            sample_ratio=args.sample,
            folds=args.folds,
            max_params=args.max_params
        )