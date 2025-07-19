"""
モデル定義・学習・予測機能
- LightGBMベースラインモデル
- クロスバリデーション対応
- ハイパーパラメータ管理
- モデル保存・読み込み機能
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from .validation import (
    CVStrategy, 
    calculate_accuracy, 
    calculate_auc, 
    calculate_prediction_distribution,
    aggregate_cv_scores,
    check_data_integrity,
    CVLogger
)

logger = logging.getLogger(__name__)


# LightGBM default parameters matching design specification
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}


class LightGBMModel:
    """LightGBM model wrapper with training and prediction functionality"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM model
        
        Args:
            params: Model hyperparameters (uses defaults if None)
        """
        self.params = params or LIGHTGBM_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
        # Validate parameters
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Validate model parameters"""
        required_params = ['objective', 'random_state']
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"Required parameter '{param}' missing")
        
        # Validate learning rate
        if 'learning_rate' in self.params:
            lr = self.params['learning_rate']
            if not (0 < lr <= 1):
                raise ValueError(f"Learning rate must be in (0, 1], got {lr}")
        
        # Validate num_leaves
        if 'num_leaves' in self.params:
            num_leaves = self.params['num_leaves']
            if num_leaves <= 0:
                raise ValueError(f"num_leaves must be positive, got {num_leaves}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'LightGBMModel':
        """
        Train the model
        
        Args:
            X: Training features
            y: Training targets
            feature_names: Optional feature names
            
        Returns:
            Self for method chaining
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Create and train model
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"LightGBM model trained with {len(self.feature_names)} features")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict on
            
        Returns:
            Binary predictions (0/1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            importance_type: Type of importance ('gain', 'split', etc.)
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance_values = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'LightGBMModel':
        """
        Load model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded LightGBMModel instance
        """
        model_data = joblib.load(filepath)
        
        instance = cls(params=model_data['params'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return instance


class CrossValidationTrainer:
    """Cross-validation trainer for model evaluation"""
    
    def __init__(self, cv_strategy: Optional[CVStrategy] = None):
        """
        Initialize CV trainer
        
        Args:
            cv_strategy: CV strategy (uses default if None)
        """
        self.cv_strategy = cv_strategy or CVStrategy()
        self.logger = CVLogger()
    
    def train_cv(
        self, 
        model_class: type,
        X: np.ndarray, 
        y: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation training
        
        Args:
            model_class: Model class to use (e.g., LightGBMModel)
            X: Features
            y: Targets
            model_params: Model parameters
            feature_names: Feature names
            
        Returns:
            CV results dictionary
        """
        start_time = time.time()
        
        # Data integrity checks
        integrity_checks = check_data_integrity(X, y)
        if not all(integrity_checks.values()):
            failed_checks = [k for k, v in integrity_checks.items() if not v]
            raise ValueError(f"Data integrity checks failed: {failed_checks}")
        
        # Initialize results
        fold_scores = []
        fold_auc_scores = []
        oof_predictions = np.zeros(len(y))
        feature_importance_list = []
        models = []
        
        # Get CV splits
        cv_splits = self.cv_strategy.split(X, y)
        
        logger.info(f"Starting {len(cv_splits)}-fold cross-validation")
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            fold_start_time = time.time()
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = model_class(params=model_params)
            model.fit(X_train, y_train, feature_names=feature_names)
            
            # Predictions
            val_pred = model.predict(X_val)
            val_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Store OOF predictions
            oof_predictions[val_idx] = val_pred_proba
            
            # Calculate scores
            fold_accuracy = calculate_accuracy(y_val, val_pred)
            fold_auc = calculate_auc(y_val, val_pred_proba)
            
            fold_scores.append(fold_accuracy)
            fold_auc_scores.append(fold_auc)
            
            # Feature importance
            if hasattr(model, 'get_feature_importance'):
                feature_importance_list.append(model.get_feature_importance())
            
            # Store model
            models.append(model)
            
            fold_time = time.time() - fold_start_time
            logger.info(
                f"Fold {fold_idx + 1}/{len(cv_splits)}: "
                f"Accuracy={fold_accuracy:.4f}, AUC={fold_auc:.4f}, "
                f"Time={fold_time:.1f}s"
            )
        
        # Aggregate results
        total_time = time.time() - start_time
        aggregated_scores = aggregate_cv_scores(fold_scores)
        aggregated_auc = aggregate_cv_scores(fold_auc_scores)
        
        # Feature importance aggregation
        feature_importance = None
        if feature_importance_list:
            feature_importance = self._aggregate_feature_importance(feature_importance_list)
        
        # Prediction distribution
        oof_binary_pred = (oof_predictions > 0.5).astype(int)
        pred_distribution = calculate_prediction_distribution(oof_binary_pred)
        
        # Create results dictionary
        cv_results = {
            'fold_scores': fold_scores,
            'fold_auc_scores': fold_auc_scores,
            'mean_score': aggregated_scores['mean_score'],
            'std_score': aggregated_scores['std_score'],
            'mean_auc': aggregated_auc['mean_score'],
            'std_auc': aggregated_auc['std_score'],
            'oof_predictions': oof_predictions,
            'oof_binary_predictions': oof_binary_pred,
            'prediction_distribution': pred_distribution,
            'feature_importance': feature_importance,
            'training_time': total_time,
            'models': models,
            'cv_config': self.cv_strategy.get_config(),
            'data_integrity_checks': integrity_checks
        }
        
        logger.info(
            f"CV completed: Accuracy={aggregated_scores['mean_score']:.4f}±"
            f"{aggregated_scores['std_score']:.4f}, "
            f"AUC={aggregated_auc['mean_score']:.4f}±"
            f"{aggregated_auc['std_score']:.4f}, "
            f"Time={total_time:.1f}s"
        )
        
        return cv_results
    
    def _aggregate_feature_importance(self, importance_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate feature importance across folds
        
        Args:
            importance_list: List of feature importance DataFrames
            
        Returns:
            Aggregated feature importance
        """
        if not importance_list:
            return None
        
        # Combine all importance dataframes
        all_importance = pd.concat(importance_list, ignore_index=True)
        
        # Calculate mean importance for each feature
        aggregated = all_importance.groupby('feature')['importance'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        aggregated.columns = ['feature', 'importance_mean', 'importance_std', 'fold_count']
        aggregated = aggregated.sort_values('importance_mean', ascending=False)
        
        return aggregated


def evaluate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive model evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'auc': calculate_auc(y_true, y_pred_proba)
    }
    
    return metrics


def create_learning_curve_data(train_scores: List[float], val_scores: List[float]) -> Dict[str, List[float]]:
    """
    Create learning curve data structure
    
    Args:
        train_scores: Training scores by epoch
        val_scores: Validation scores by epoch
        
    Returns:
        Learning curve data
    """
    return {
        'train_scores': train_scores,
        'val_scores': val_scores,
        'epochs': list(range(len(train_scores)))
    }


def save_model_with_metadata(
    model: LightGBMModel, 
    cv_results: Dict[str, Any], 
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model with comprehensive metadata
    
    Args:
        model: Trained model
        cv_results: Cross-validation results
        filepath: Path to save the model
        metadata: Additional metadata
    """
    model_data = {
        'model': model.model,
        'params': model.params,
        'feature_names': model.feature_names,
        'cv_score': cv_results['mean_score'],
        'cv_std': cv_results['std_score'],
        'cv_results': cv_results,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'LightGBM',
        'version': '1.0.0'
    }
    
    if metadata:
        model_data.update(metadata)
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_data, filepath)
    logger.info(f"Model with metadata saved to {filepath}")


def load_model_with_metadata(filepath: str) -> Tuple[LightGBMModel, Dict[str, Any]]:
    """
    Load model with metadata
    
    Args:
        filepath: Path to load the model from
        
    Returns:
        Tuple of (model, metadata)
    """
    data = joblib.load(filepath)
    
    # Reconstruct model
    model = LightGBMModel(params=data['params'])
    model.model = data['model']
    model.feature_names = data['feature_names']
    model.is_fitted = True
    
    # Extract metadata
    metadata = {k: v for k, v in data.items() if k not in ['model', 'params', 'feature_names']}
    
    logger.info(f"Model with metadata loaded from {filepath}")
    return model, metadata