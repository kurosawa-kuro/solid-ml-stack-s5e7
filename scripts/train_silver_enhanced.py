#!/usr/bin/env python
"""
Enhanced Silver Layer Training Script
Implements top 3 high-impact strategies for Bronze medal achievement
Expected improvement: +0.8% (from 0.9684 to 0.976518)
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
from src.data.gold import prepare_data
from src.models import LightGBMModel
from src.util.notifications import NotificationManager
from src.util.time_tracker import TimeTracker
from src.validation import cross_validate

warnings.filterwarnings("ignore")


def main():
    """Main training pipeline with enhanced Silver features"""
    tracker = TimeTracker()
    notifier = NotificationManager()
    
    print("=" * 80)
    print("Enhanced Silver Layer Training - Bronze Medal Target")
    print("=" * 80)
    print(f"Target Score: 0.976518 (Bronze Medal)")
    print(f"Current Best: 0.9684")
    print(f"Gap to Close: +0.008 (0.8%)")
    print("=" * 80)
    
    try:
        # Step 1: Ensure Bronze tables exist
        tracker.start("bronze_setup")
        print("\n1. Setting up Bronze layer...")
        try:
            train_bronze, test_bronze = load_data()
            print(f"   ✓ Bronze data loaded: {train_bronze.shape}")
        except:
            print("   - Creating Bronze tables...")
            create_bronze_tables()
            train_bronze, test_bronze = load_data()
            print(f"   ✓ Bronze tables created: {train_bronze.shape}")
        tracker.end("bronze_setup")
        
        # Step 2: Prepare data for Gold layer
        tracker.start("data_preparation")
        print("\n2. Preparing data with enhanced Silver features...")
        
        # Get target before transformations
        X_train, y_train, X_test = prepare_data(train_bronze, test_bronze)
        print(f"   ✓ Initial shapes - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Apply enhanced Silver transformations
        print("\n3. Applying enhanced Silver transformations...")
        print("   - Strategy 1: CatBoost-specific features (binning, clustering, power transforms)")
        print("   - Strategy 2: Target encoding with CV safety")
        print("   - Strategy 3: Advanced statistical features + KNN imputation")
        
        # Apply transformations with potential SMOTE
        result = apply_enhanced_silver_features(X_train, y_train, is_train=True)
        if isinstance(result, tuple):
            X_train_enhanced, y_train_enhanced = result
            print(f"   ✓ SMOTE applied - new shape: {X_train_enhanced.shape}")
        else:
            X_train_enhanced = result
            y_train_enhanced = y_train
        
        # Apply to test set (no target encoding or SMOTE)
        X_test_enhanced = apply_enhanced_silver_features(X_test, is_train=False)
        
        print(f"   ✓ Enhanced shapes - Train: {X_train_enhanced.shape}, Test: {X_test_enhanced.shape}")
        print(f"   ✓ New features added: {X_train_enhanced.shape[1] - X_train.shape[1]}")
        tracker.end("data_preparation")
        
        # Step 3: Feature selection based on importance
        tracker.start("feature_selection")
        print("\n4. Selecting most important features...")
        
        # Quick training to get feature importance
        quick_model = LightGBMModel(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        quick_model.fit(X_train_enhanced, y_train_enhanced)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X_train_enhanced.columns,
            'importance': quick_model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        n_features = min(100, len(importance_df))  # Use top 100 features
        top_features = importance_df.head(n_features)['feature'].tolist()
        
        X_train_selected = X_train_enhanced[top_features]
        X_test_selected = X_test_enhanced[top_features]
        
        print(f"   ✓ Selected {len(top_features)} features from {X_train_enhanced.shape[1]}")
        print(f"   ✓ Top 10 features:")
        for i, (feat, imp) in enumerate(importance_df.head(10).values):
            print(f"      {i+1}. {feat}: {imp:.1f}")
        tracker.end("feature_selection")
        
        # Step 4: Cross-validation with optimized parameters
        tracker.start("cross_validation")
        print("\n5. Running cross-validation with enhanced features...")
        
        # Optimized parameters for Bronze medal
        model_params = {
            "n_estimators": 200,
            "num_leaves": 50,
            "learning_rate": 0.08,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 15,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        
        model = LightGBMModel(**model_params)
        cv_results = cross_validate(
            model, 
            X_train_selected, 
            y_train_enhanced,
            cv_folds=5,
            stratified=True,
            return_predictions=True
        )
        
        print(f"\n   ✓ CV Results:")
        print(f"     - Mean Score: {cv_results['mean_score']:.6f}")
        print(f"     - Std Score:  {cv_results['std_score']:.6f}")
        print(f"     - Gap to Bronze: {0.976518 - cv_results['mean_score']:+.6f}")
        
        tracker.end("cross_validation")
        
        # Step 5: Train final model and make predictions
        if cv_results['mean_score'] >= 0.975:  # Close to Bronze
            tracker.start("final_training")
            print("\n6. Training final model for submission...")
            
            model.fit(X_train_selected, y_train_enhanced)
            predictions = model.predict(X_test_selected)
            
            # Create submission
            submission = pd.DataFrame({
                "id": test_bronze["id"],
                "Personality": predictions
            })
            
            submission_path = "submissions/enhanced_silver_submission.csv"
            submission.to_csv(submission_path, index=False)
            print(f"   ✓ Submission saved to {submission_path}")
            
            tracker.end("final_training")
        
        # Summary
        total_time = tracker.get_total_time()
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Total Time: {total_time:.2f}s")
        print(f"CV Score: {cv_results['mean_score']:.6f} ± {cv_results['std_score']:.6f}")
        print(f"Bronze Medal Gap: {0.976518 - cv_results['mean_score']:+.6f}")
        print("=" * 80)
        
        # Feature contribution analysis
        print("\nFeature Type Contribution:")
        feature_types = {
            'binned': len([f for f in top_features if '_binned' in f]),
            'cluster': len([f for f in top_features if 'cluster_' in f]),
            'power': len([f for f in top_features if '_power' in f]),
            'target_encoded': len([f for f in top_features if '_target_encoded' in f]),
            'statistical': len([f for f in top_features if any(s in f for s in ['_mean', '_std', '_skew', '_kurtosis', '_percentile'])]),
            'original': len([f for f in top_features if not any(s in f for s in ['_binned', 'cluster_', '_power', '_target_encoded', '_mean', '_std', '_skew', '_kurtosis', '_percentile'])])
        }
        
        print("\nTop features by type:")
        for ftype, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  - {ftype}: {count} features")
        
        # Notify completion
        if cv_results['mean_score'] >= 0.976:
            notifier.notify(
                "🏆 Bronze Medal Achieved!",
                f"CV Score: {cv_results['mean_score']:.6f}",
                level="success"
            )
        elif cv_results['mean_score'] >= 0.975:
            notifier.notify(
                "🎯 Very Close to Bronze!",
                f"CV Score: {cv_results['mean_score']:.6f} (Gap: {0.976518 - cv_results['mean_score']:.6f})",
                level="warning"
            )
        else:
            notifier.notify(
                "Training Complete",
                f"CV Score: {cv_results['mean_score']:.6f}",
                level="info"
            )
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        notifier.notify("Training Failed", str(e), level="error")
        sys.exit(1)


if __name__ == "__main__":
    main()