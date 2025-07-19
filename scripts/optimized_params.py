#!/usr/bin/env python
"""
Optimized LightGBM parameters for immediate performance boost
Bronze medal target: 0.976518
"""

def get_baseline_params():
    """Current baseline parameters"""
    return {
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


def get_optimized_params_v1():
    """Optimized parameters - Version 1: Balanced approach"""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        
        # Model complexity (increased for better performance)
        "n_estimators": 300,        # 100→300: More iterations
        "num_leaves": 100,          # 31→100: More complex trees
        "max_depth": 8,             # Control overfitting while allowing complexity
        
        # Learning rate (reduced for stability)
        "learning_rate": 0.05,      # 0.1→0.05: More careful learning
        
        # Feature sampling (increased for better coverage)
        "feature_fraction": 0.9,    # 0.8→0.9: Use more features
        "bagging_fraction": 0.85,   # 0.8→0.85: Slightly more data per tree
        "bagging_freq": 5,
        
        # Regularization (added for generalization)
        "lambda_l1": 0.1,           # L1 regularization
        "lambda_l2": 0.1,           # L2 regularization
        "min_data_in_leaf": 5,      # 20→5: Allow smaller leaf nodes
        "min_gain_to_split": 0.1,   # Minimum gain for splits
        
        # Performance
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "force_row_wise": True,     # Better for small datasets
    }


def get_optimized_params_v2():
    """Optimized parameters - Version 2: Aggressive approach"""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        
        # More aggressive complexity
        "n_estimators": 500,        # Even more iterations
        "num_leaves": 150,          # Higher complexity
        "max_depth": 10,
        
        # Conservative learning
        "learning_rate": 0.03,      # Even slower learning
        
        # Maximum feature utilization
        "feature_fraction": 0.95,   # Use almost all features
        "bagging_fraction": 0.9,    
        "bagging_freq": 3,          # More frequent bagging
        
        # Strong regularization
        "lambda_l1": 0.2,
        "lambda_l2": 0.2,
        "min_data_in_leaf": 3,      # Very fine-grained splits
        "min_gain_to_split": 0.05,
        
        # Performance
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "force_row_wise": True,
    }


def get_optimized_params_v3():
    """Optimized parameters - Version 3: Speed vs Performance balance"""
    return {
        "objective": "binary",
        "metric": "binary_logloss", 
        "boosting_type": "gbdt",
        
        # Moderate complexity for speed
        "n_estimators": 200,        # Good balance
        "num_leaves": 80,           # Reasonable complexity
        "max_depth": 7,
        
        # Moderate learning rate
        "learning_rate": 0.07,      # Balanced speed vs accuracy
        
        # Good feature sampling
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        
        # Moderate regularization
        "lambda_l1": 0.05,
        "lambda_l2": 0.05,
        "min_data_in_leaf": 8,
        "min_gain_to_split": 0.1,
        
        # Additional optimization
        "cat_smooth": 10,           # For categorical features
        "max_cat_threshold": 32,
        
        # Performance
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "force_row_wise": True,
    }


def get_competition_tuned_params():
    """Competition-specific tuned parameters based on data characteristics"""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        
        # Optimized for personality prediction task
        "n_estimators": 350,        # Sweet spot for this dataset size
        "num_leaves": 120,          # Good for 473 features
        "max_depth": 9,             # Deep enough for interactions
        
        # Tuned learning rate
        "learning_rate": 0.04,      # Slow and steady
        
        # Feature optimization (473 features available)
        "feature_fraction": 0.92,   # Use most features but not all
        "bagging_fraction": 0.88,   # Good generalization
        "bagging_freq": 4,
        
        # Regularization tuned for this problem
        "lambda_l1": 0.15,          # Prevent overfitting on many features
        "lambda_l2": 0.15,
        "min_data_in_leaf": 6,      # Small enough for pattern detection
        "min_gain_to_split": 0.08,
        
        # Categorical handling (for encoded features)
        "cat_smooth": 15,
        "max_cat_threshold": 64,
        
        # Extra boosting parameters
        "extra_trees": False,
        "path_smooth": 0.1,
        
        # Performance
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "force_row_wise": True,
        "deterministic": True,      # For reproducibility
    }


# Parameter sets for quick testing
PARAM_SETS = {
    "baseline": get_baseline_params(),
    "optimized_v1": get_optimized_params_v1(),
    "optimized_v2": get_optimized_params_v2(), 
    "optimized_v3": get_optimized_params_v3(),
    "competition_tuned": get_competition_tuned_params(),
}


def get_best_params_for_target(target="bronze_medal"):
    """Get the best parameter set for specific target"""
    if target == "bronze_medal":
        return get_competition_tuned_params()
    elif target == "speed":
        return get_optimized_params_v3()
    elif target == "accuracy":
        return get_optimized_params_v2()
    else:
        return get_optimized_params_v1()


if __name__ == "__main__":
    print("Available parameter sets:")
    for name, params in PARAM_SETS.items():
        print(f"\n{name}:")
        print(f"  n_estimators: {params['n_estimators']}")
        print(f"  num_leaves: {params['num_leaves']}")
        print(f"  learning_rate: {params['learning_rate']}")
        print(f"  regularization: L1={params.get('lambda_l1', 0)}, L2={params.get('lambda_l2', 0)}")