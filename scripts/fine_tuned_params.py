#!/usr/bin/env python
"""
Fine-tuned parameters based on benchmark results
Baseline performed best, so making incremental improvements
"""

def get_fine_tuned_v1():
    """Fine-tuned Version 1: Incremental improvement on baseline"""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        
        # Slight increase from baseline
        "n_estimators": 150,        # 100→150: More iterations
        "num_leaves": 40,           # 31→40: Slightly more complexity
        "learning_rate": 0.08,      # 0.1→0.08: Slightly more careful
        
        # Keep successful baseline settings
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        
        # Add minimal regularization
        "lambda_l1": 0.01,          # Very light L1
        "lambda_l2": 0.01,          # Very light L2
        
        # Performance
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }


def get_fine_tuned_v2():
    """Fine-tuned Version 2: Focus on feature handling"""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        
        # Moderate increase for 481 features
        "n_estimators": 200,        # Good for feature-rich dataset
        "num_leaves": 50,           # Balance complexity vs overfitting
        "learning_rate": 0.06,      # Slower learning for stability
        
        # Optimized for many features (481)
        "feature_fraction": 0.7,    # Sample features to prevent overfitting
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 15,    # Slightly lower for fine patterns
        
        # Light regularization
        "lambda_l1": 0.05,
        "lambda_l2": 0.05,
        "min_gain_to_split": 0.01,  # Allow more splits
        
        # Performance
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "force_row_wise": True,
    }


def get_fine_tuned_v3():
    """Fine-tuned Version 3: Early stopping approach"""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        
        # Higher iterations with early stopping intent
        "n_estimators": 1000,       # High ceiling
        "num_leaves": 45,           # Conservative complexity
        "learning_rate": 0.05,      # Slow learning
        
        # Conservative sampling for stability
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_child_samples": 18,
        
        # Balanced regularization
        "lambda_l1": 0.02,
        "lambda_l2": 0.02,
        
        # Early stopping (would need validation set)
        "min_gain_to_split": 0.005,
        
        # Performance
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "force_row_wise": True,
    }


def get_ensemble_params():
    """Parameters designed for ensemble averaging"""
    base_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 120,
        "num_leaves": 35,
        "learning_rate": 0.07,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbosity": -1,
        "n_jobs": -1,
    }
    
    # Return multiple variants for ensemble
    variants = []
    for i, seed in enumerate([42, 123, 456, 789, 999]):
        params = base_params.copy()
        params["random_state"] = seed
        # Slight variations
        params["feature_fraction"] = 0.8 - 0.05 * (i % 3)  # 0.8, 0.75, 0.7
        params["bagging_fraction"] = 0.8 + 0.05 * (i % 2)  # 0.8, 0.85
        variants.append(params)
    
    return variants


# Updated parameter sets including fine-tuned versions
FINE_TUNED_PARAMS = {
    "baseline": {
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
    },
    "fine_tuned_v1": get_fine_tuned_v1(),
    "fine_tuned_v2": get_fine_tuned_v2(),
    "fine_tuned_v3": get_fine_tuned_v3(),
}


if __name__ == "__main__":
    print("Fine-tuned parameter sets:")
    for name, params in FINE_TUNED_PARAMS.items():
        print(f"\n{name}:")
        print(f"  n_estimators: {params['n_estimators']}")
        print(f"  num_leaves: {params['num_leaves']}")
        print(f"  learning_rate: {params['learning_rate']}")
        print(f"  feature_fraction: {params['feature_fraction']}")
        print(f"  regularization: L1={params.get('lambda_l1', 0)}, L2={params.get('lambda_l2', 0)}")
    
    print(f"\nEnsemble variants: {len(get_ensemble_params())} models")