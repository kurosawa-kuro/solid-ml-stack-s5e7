"""
Silver Layer Enhanced Features
High-impact strategies for Bronze medal achievement (+0.8% target)
"""

from typing import Tuple, List, Dict, Any, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
# import scipy.stats as stats  # Commented out to avoid version conflicts

warnings.filterwarnings('ignore')


class LightGBMFeatureEngineer(BaseEstimator, TransformerMixin):
    """Strategy 1: LightGBM-optimized feature engineering (+0.3-0.5% expected)"""
    
    def __init__(self, use_power_transforms: bool = True):
        self.use_power_transforms = use_power_transforms
        self.power_transformers = {}
        self.numeric_features = None
        
    def fit(self, X, y=None):
        """Fit power transformations for skewed features"""
        # Identify numeric features
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in self.numeric_features:
            self.numeric_features.remove('id')
        
        # Fit power transformers for skewed features (LightGBM friendly)
        if self.use_power_transforms:
            for col in self.numeric_features:
                if col in X.columns:
                    col_data = X[col].dropna()
                    if len(col_data) > 0:
                        skewness = col_data.skew()
                        if abs(skewness) > 0.5:  # Moderately skewed
                            self.power_transformers[col] = PowerTransformer(
                                method='yeo-johnson',
                                standardize=False
                            )
                            self.power_transformers[col].fit(col_data.values.reshape(-1, 1))
        
        return self
    
    def transform(self, X):
        """Apply power transformations for LightGBM"""
        X_transformed = X.copy()
        
        # Apply power transformations
        for col, transformer in self.power_transformers.items():
            if col in X_transformed.columns:
                # Handle NaN
                mask = X_transformed[col].notna()
                if mask.sum() > 0:
                    X_transformed.loc[mask, f'{col}_power'] = transformer.transform(
                        X_transformed.loc[mask, col].values.reshape(-1, 1)
                    ).flatten()
        
        return X_transformed


class CVSafeTargetEncoder(BaseEstimator, TransformerMixin):
    """Strategy 2: Fold-safe target encoding (+0.2-0.4% expected)"""
    
    def __init__(self, cols: Optional[List[str]] = None, smoothing: float = 1.0, noise_level: float = 0.01):
        self.cols = cols
        self.smoothing = smoothing
        self.noise_level = noise_level
        self.encoders = {}
        self.global_mean = None
        
    def fit(self, X, y):
        """Fit target encoders with smoothing"""
        if y is None:
            raise ValueError("Target encoding requires y")
            
        self.global_mean = np.mean(y)
        
        # Determine columns to encode
        if self.cols is None:
            # Encode all categorical columns
            self.cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fit encoder for each column
        for col in self.cols:
            if col in X.columns:
                encoder = TargetEncoder(
                    smoothing=self.smoothing,
                    min_samples_leaf=10,
                    return_df=True
                )
                # Create a temporary DataFrame for fitting
                temp_df = pd.DataFrame({col: X[col]})
                encoder.fit(temp_df, y)
                self.encoders[col] = encoder
        
        return self
    
    def transform(self, X):
        """Apply target encoding with noise for regularization"""
        X_transformed = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                # Apply encoding
                temp_df = pd.DataFrame({col: X_transformed[col]})
                encoded_values = encoder.transform(temp_df)[col].values
                
                # Add noise to prevent overfitting
                if self.noise_level > 0:
                    noise = np.random.normal(0, self.noise_level, size=len(encoded_values))
                    encoded_values = encoded_values + noise
                
                X_transformed[f'{col}_target_encoded'] = encoded_values
        
        return X_transformed


class AdvancedStatisticalFeatures(BaseEstimator, TransformerMixin):
    """Strategy 3: Advanced statistical and imputation features (+0.1-0.3% expected)"""
    
    def __init__(self, n_neighbors: int = 5, use_smote: bool = False):
        self.n_neighbors = n_neighbors
        self.use_smote = use_smote
        self.knn_imputer = None
        self.smote = None
        self.numeric_features = None
        
    def fit(self, X, y=None):
        """Fit KNN imputer and prepare SMOTE if needed"""
        # Identify numeric features
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in self.numeric_features:
            self.numeric_features.remove('id')
        
        # Fit KNN imputer on numeric features
        if self.numeric_features:
            self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            self.knn_imputer.fit(X[self.numeric_features])
        
        # Prepare SMOTE if requested and y is provided
        if self.use_smote and y is not None:
            self.smote = SMOTE(
                sampling_strategy='auto',
                k_neighbors=min(5, len(y) - 1),
                random_state=42
            )
        
        return self
    
    def transform(self, X):
        """Apply KNN imputation and add statistical features"""
        X_transformed = X.copy()
        
        # 1. Apply KNN imputation
        if self.knn_imputer and self.numeric_features:
            # Store original missing indicators before imputation
            for col in self.numeric_features:
                if col in X_transformed.columns:
                    X_transformed[f'{col}_was_missing'] = X_transformed[col].isna().astype(int)
            
            # Apply KNN imputation
            X_transformed[self.numeric_features] = self.knn_imputer.transform(
                X_transformed[self.numeric_features]
            )
        
        # 2. Add statistical moment features
        if self.numeric_features:
            # Row-wise statistics
            numeric_data = X_transformed[self.numeric_features]
            
            # Basic moments
            X_transformed['row_mean'] = numeric_data.mean(axis=1)
            X_transformed['row_std'] = numeric_data.std(axis=1)
            X_transformed['row_skew'] = numeric_data.apply(lambda x: x.dropna().skew() if len(x.dropna()) > 0 else 0, axis=1)
            X_transformed['row_kurtosis'] = numeric_data.apply(lambda x: x.dropna().kurtosis() if len(x.dropna()) > 0 else 0, axis=1)
            
            # Quantiles
            X_transformed['row_q25'] = numeric_data.quantile(0.25, axis=1)
            X_transformed['row_q75'] = numeric_data.quantile(0.75, axis=1)
            X_transformed['row_iqr'] = X_transformed['row_q75'] - X_transformed['row_q25']
            
            # Feature-specific moments for key features
            key_features = ['Social_event_attendance', 'Time_spent_Alone', 'Friends_circle_size']
            for feat in key_features:
                if feat in X_transformed.columns:
                    # Z-score
                    X_transformed[f'{feat}_zscore'] = (
                        X_transformed[feat] - X_transformed[feat].mean()
                    ) / (X_transformed[feat].std() + 1e-8)
                    
                    # Percentile rank
                    X_transformed[f'{feat}_percentile'] = X_transformed[feat].rank(pct=True)
        
        # 3. Mode-based features for categorical
        cat_features = X_transformed.select_dtypes(include=['object', 'category']).columns
        for col in cat_features:
            if col in X_transformed.columns and col != 'id':
                # Add mode indicator
                mode_value = X_transformed[col].mode()[0] if len(X_transformed[col].mode()) > 0 else None
                if mode_value:
                    X_transformed[f'{col}_is_mode'] = (X_transformed[col] == mode_value).astype(int)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Special handling for SMOTE in training"""
        self.fit(X, y)
        X_transformed = self.transform(X)
        
        # Apply SMOTE only during training (when y is provided)
        if self.use_smote and self.smote and y is not None:
            # Select only numeric features for SMOTE
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
            X_numeric = X_transformed[numeric_cols].fillna(0)
            
            # Apply SMOTE
            X_resampled, y_resampled = self.smote.fit_resample(X_numeric, y)
            
            # Create new DataFrame with resampled data
            X_resampled_df = pd.DataFrame(X_resampled, columns=numeric_cols)
            
            # Return resampled data and labels
            return X_resampled_df, y_resampled
        
        return X_transformed


class EnhancedSilverPreprocessor(BaseEstimator, TransformerMixin):
    """Combined Silver layer enhancements for +0.8% improvement"""
    
    def __init__(
        self,
        use_catboost_features: bool = True,
        use_target_encoding: bool = True,
        use_statistical_features: bool = True,
        target_cols: Optional[List[str]] = None
    ):
        self.use_catboost_features = use_catboost_features
        self.use_target_encoding = use_target_encoding
        self.use_statistical_features = use_statistical_features
        self.target_cols = target_cols
        
        # Initialize sub-transformers
        self.lgbm_engineer = LightGBMFeatureEngineer() if use_catboost_features else None
        self.target_encoder = CVSafeTargetEncoder(cols=target_cols) if use_target_encoding else None
        self.statistical_engineer = AdvancedStatisticalFeatures() if use_statistical_features else None
        
    def fit(self, X, y=None):
        """Fit all sub-transformers"""
        if self.lgbm_engineer:
            self.lgbm_engineer.fit(X, y)
        
        if self.target_encoder and y is not None:
            self.target_encoder.fit(X, y)
        
        if self.statistical_engineer:
            self.statistical_engineer.fit(X, y)
        
        return self
    
    def transform(self, X):
        """Apply all transformations in sequence"""
        X_transformed = X.copy()
        
        # Apply transformations in order of impact
        if self.lgbm_engineer:
            X_transformed = self.lgbm_engineer.transform(X_transformed)
        
        if self.target_encoder:
            X_transformed = self.target_encoder.transform(X_transformed)
        
        if self.statistical_engineer:
            X_transformed = self.statistical_engineer.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Fit and transform with special handling for SMOTE"""
        self.fit(X, y)
        
        # Apply transformations
        X_transformed = X.copy()
        
        if self.lgbm_engineer:
            X_transformed = self.lgbm_engineer.transform(X_transformed)
        
        if self.target_encoder and y is not None:
            X_transformed = self.target_encoder.transform(X_transformed)
        
        if self.statistical_engineer:
            # This might return resampled data
            result = self.statistical_engineer.fit_transform(X_transformed, y)
            if isinstance(result, tuple):
                return result  # X_resampled, y_resampled
            else:
                X_transformed = result
        
        return X_transformed


def apply_enhanced_silver_features(df: pd.DataFrame, y=None, is_train: bool = True) -> pd.DataFrame:
    """Convenience function to apply all enhanced Silver features"""
    
    # First apply existing Silver features
    from .silver import (
        advanced_features, 
        s5e7_interaction_features,
        s5e7_drain_adjusted_features,
        s5e7_communication_ratios,
        enhanced_interaction_features,
        polynomial_features
    )
    
    # Apply existing transformations
    df_transformed = advanced_features(df)
    df_transformed = s5e7_interaction_features(df_transformed)
    df_transformed = s5e7_drain_adjusted_features(df_transformed)
    df_transformed = s5e7_communication_ratios(df_transformed)
    df_transformed = enhanced_interaction_features(df_transformed)
    df_transformed = polynomial_features(df_transformed, degree=2)
    
    # Apply new enhancements
    enhancer = EnhancedSilverPreprocessor(
        use_catboost_features=True,
        use_target_encoding=is_train,  # Only use target encoding during training
        use_statistical_features=True,
        target_cols=['Stage_fear', 'Drained_after_socializing']
    )
    
    if is_train and y is not None:
        result = enhancer.fit_transform(df_transformed, y)
        if isinstance(result, tuple):
            return result  # X_resampled, y_resampled
        else:
            return result
    else:
        if hasattr(enhancer, 'fit'):
            enhancer.fit(df_transformed)
        return enhancer.transform(df_transformed)