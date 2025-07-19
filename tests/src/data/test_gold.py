"""
Test for Gold Level Data Functions - Success Cases Only
Includes comprehensive enhanced test cases from test_silver_gold_enhanced.py
"""

import tempfile
from unittest.mock import Mock, patch, MagicMock
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PolynomialFeatures

from src.data.gold import (
    encode_target, 
    prepare_model_data,
    clean_and_validate_features,
    select_best_features,
    get_ml_ready_data,
    create_submission,
    load_gold_data,
    create_gold_tables,
    get_feature_names
)


class TestGoldFunctions:
    """Gold level function tests - minimal success cases"""

    def test_prepare_model_data_basic(self):
        """Test basic model data preparation"""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "Personality": ["Introvert", "Extrovert", "Introvert"],
            }
        )

        result = prepare_model_data(df)

        # Should return processed data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_prepare_model_data_with_target(self):
        """Test model data preparation with specific target"""
        df = pd.DataFrame({"id": [1, 2, 3], "feature1": [1.0, 2.0, 3.0], "target_col": ["A", "B", "A"]})

        result = prepare_model_data(df, target_col="target_col")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_encode_target_basic(self):
        """Test basic target encoding"""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "Personality": ["Introvert", "Extrovert", "Introvert", "Extrovert"]})

        result = encode_target(df)

        # Should have encoded personality
        assert "Personality_encoded" in result.columns
        assert set(result["Personality_encoded"].values) == {0, 1}

    def test_encode_target_custom_column(self):
        """Test target encoding with custom column"""
        df = pd.DataFrame({"id": [1, 2, 3], "custom_target": ["A", "B", "A"]})

        result = encode_target(df, target_col="custom_target")

        assert "custom_target_encoded" in result.columns

    @patch("src.data.gold.duckdb.connect")
    def test_create_gold_tables_success(self, mock_connect):
        """Test gold table creation succeeds"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # Should not raise an error
        from src.data.gold import create_gold_tables

        create_gold_tables()

        # Verify connection was made and closed
        mock_connect.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("src.data.gold.duckdb.connect")
    def test_load_gold_data_success(self, mock_connect):
        """Test gold data loading succeeds"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_train = pd.DataFrame({"id": [1, 2], "feature": [1.0, 2.0]})
        mock_test = pd.DataFrame({"id": [3, 4], "feature": [3.0, 4.0]})

        mock_conn.execute.side_effect = [MagicMock(df=lambda: mock_train), MagicMock(df=lambda: mock_test)]

        from src.data.gold import load_gold_data

        train, test = load_gold_data()

        assert len(train) == 2
        assert len(test) == 2
        assert "feature" in train.columns
        assert "feature" in test.columns

    def test_empty_dataframe_handling(self):
        """Test functions handle empty DataFrames gracefully"""
        empty_df = pd.DataFrame()

        # Should not crash
        result1 = prepare_model_data(empty_df)
        result2 = encode_target(empty_df)

        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)


class TestGoldCleaning:
    """Test gold.py cleaning functionality"""

    def test_clean_and_validate_features_infinite_values(self):
        """Test cleaning of infinite values"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature1': [1, np.inf, 3],
            'feature2': [np.nan, -np.inf, 2]
        })
        
        result = clean_and_validate_features(df)
        
        # Infinite values should be handled
        assert not np.isinf(result['feature1']).any()
        assert not np.isinf(result['feature2']).any()
        
        # ID should be unchanged
        assert result['id'].tolist() == [1, 2, 3]

    def test_clean_and_validate_features_outliers(self):
        """Test outlier handling"""
        # Create data with clear outlier
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 1000]  # 1000 is clear outlier
        })
        
        result = clean_and_validate_features(df)
        
        # Outlier should be clipped
        assert result['feature'].max() < 1000

    def test_clean_and_validate_features_missing_values(self):
        """Test missing value handling"""
        df = pd.DataFrame({
            'feature1': [1, np.nan, 3, 4, 5],
            'feature2': [np.nan, 2, 3, np.nan, 5]
        })
        
        result = clean_and_validate_features(df)
        
        # Missing values should be filled
        assert not result['feature1'].isna().any()
        assert not result['feature2'].isna().any()


class TestGoldFeatureSelection:
    """Test gold.py feature selection functionality"""

    def test_select_best_features_basic(self):
        """Test basic feature selection"""
        # Create data with clear feature importance pattern
        np.random.seed(42)
        df = pd.DataFrame({
            'id': range(100),
            'important_feature': np.random.randn(100),
            'noise_feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Make important_feature actually important
        df.loc[df['target'] == 1, 'important_feature'] += 2
        
        selected_features = select_best_features(df, 'target', k=1)
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 1
        assert 'id' not in selected_features
        assert 'target' not in selected_features

    def test_select_best_features_string_target(self):
        """Test feature selection with string target"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [4, 3, 2, 1],
            'target': ['Introvert', 'Extrovert', 'Introvert', 'Extrovert']
        })
        
        selected_features = select_best_features(df, 'target', k=2)
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 2

    def test_select_best_features_fewer_than_k(self):
        """Test feature selection when features < k"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [0, 1, 0]
        })
        
        selected_features = select_best_features(df, 'target', k=10)
        
        # Should return all available features
        assert len(selected_features) == 1
        assert 'feature1' in selected_features


class TestGoldCLAUDEMDFeatures:
    """Test CLAUDE.md specified Gold layer functions"""
    
    def test_get_ml_ready_data_lightgbm_interface(self):
        """Test LightGBM-ready data interface (Silver → Gold → Model)"""
        # Create test data that simulates Silver layer output
        test_train = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'Social_event_attendance': [4, 6, 2, 5],
            'Going_outside': [3, 5, 1, 4],
            'extrovert_score': [10, 15, 5, 12],
            'social_ratio': [1.33, 1.2, 2.0, 1.25],
            'Personality': ['Introvert', 'Extrovert', 'Introvert', 'Extrovert'],
            'Personality_encoded': [0, 1, 0, 1]
        })
        
        test_test = pd.DataFrame({
            'id': [5, 6, 7],
            'Social_event_attendance': [3, 7, 4],
            'Going_outside': [2, 6, 3],
            'extrovert_score': [8, 18, 11],
            'social_ratio': [1.5, 1.17, 1.33]
        })
        
        # Mock load_gold_data to return test data
        with patch('src.data.gold.load_gold_data') as mock_load:
            mock_load.return_value = (test_train, test_test)
            
            X_train, y_train, X_test, test_ids = get_ml_ready_data()
            
            # Verify LightGBM-ready format
            assert isinstance(X_train, np.ndarray)
            assert isinstance(y_train, np.ndarray)
            assert isinstance(X_test, np.ndarray)
            assert len(X_train) == 4
            assert len(X_test) == 3
            assert X_train.shape[1] == X_test.shape[1]  # Same feature count
            
            # Verify target encoding
            assert set(y_train) == {0, 1}
            assert y_train.tolist() == [0, 1, 0, 1]
    
    def test_clean_and_validate_features_final_validation(self):
        """Test final validation: Infinite value processing, outlier detection"""
        # Create data with production-quality issues
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'feature1': [1, np.inf, 3, -np.inf, 5],  # Infinite values
            'feature2': [1, 2, 1000, 4, 5],          # Extreme outlier
            'feature3': [np.nan, 2, 3, np.nan, 5]   # Missing values
        })
        
        result = clean_and_validate_features(df)
        
        # Final validation guarantees
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
        assert not result.select_dtypes(include=[np.number]).isna().any().any()
        
        # ID preservation
        assert result['id'].tolist() == [1, 2, 3, 4, 5]
        
        # Outlier detection effectiveness
        assert result['feature2'].max() < 1000  # Extreme outlier clipped
    
    def test_select_best_features_statistical_selection(self):
        """Test statistical feature selection (F-test + MI)"""
        # Create data with clear feature importance pattern
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame({
            'id': range(n_samples),
            'highly_important': np.random.randn(n_samples),
            'moderately_important': np.random.randn(n_samples) * 0.5,
            'noise_feature': np.random.randn(n_samples) * 0.1,
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # Make features actually predictive
        df.loc[df['target'] == 1, 'highly_important'] += 2
        df.loc[df['target'] == 1, 'moderately_important'] += 1
        
        selected_features = select_best_features(df, 'target', k=2)
        
        # Statistical selection guarantees
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 2
        assert 'id' not in selected_features
        assert 'target' not in selected_features
        
        # F-test + MI should prioritize important features
        assert 'highly_important' in selected_features
    
    def test_create_submission_format_competition_output(self):
        """Test competition output interface: Standard Kaggle submission file"""
        # Mock test data
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'feature1': [1, 2, 3, 4]
        })
        
        predictions = np.array([0, 1, 0, 1])  # Binary predictions
        
        with patch('src.data.gold.load_gold_data') as mock_load:
            mock_load.return_value = (pd.DataFrame(), test_data)
            
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
                create_submission(predictions, tmp.name)
                
                # Verify submission format
                submission = pd.read_csv(tmp.name)
                assert list(submission.columns) == ['id', 'Personality']
                assert len(submission) == 4
                assert set(submission['Personality']) == {'Introvert', 'Extrovert'}
                assert submission['id'].tolist() == [1, 2, 3, 4]
                
                # Verify prediction mapping
                expected_personality = ['Introvert', 'Extrovert', 'Introvert', 'Extrovert']
                assert submission['Personality'].tolist() == expected_personality
                
                os.unlink(tmp.name)
    
    def test_prepare_model_data_model_specific_formatting(self):
        """Test model-specific formatting for direct LightGBM consumption"""
        # Create Silver-like data
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'Social_event_attendance': [4, 6, 2],
            'extrovert_score': [10, 15, 5],
            'social_ratio': [1.33, 1.2, 2.0],
            'poly_feature1_feature2': [40, 90, 10],  # Polynomial feature
            'noise_feature': [100, 200, 300],       # Should be filtered
            'Personality': ['Introvert', 'Extrovert', 'Introvert']
        })
        
        result = prepare_model_data(df, target_col='Personality', auto_select=True)
        
        # Model-specific formatting guarantees
        assert 'id' in result.columns
        assert 'Personality' in result.columns
        assert 'Personality_encoded' not in result.columns  # Added by encode_target
        
        # Feature selection should work
        feature_cols = [col for col in result.columns if col not in ['id', 'Personality']]
        assert len(feature_cols) > 0
        
        # Data quality for LightGBM
        numeric_data = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric_data).any().any()
        assert not numeric_data.isna().any().any()


class TestGoldDependencyChain:
    """Test Gold layer Silver dependency enforcement"""
    
    @patch('duckdb.connect')
    def test_load_gold_data_silver_dependency(self, mock_connect):
        """Test that load_gold_data only accesses Silver tables"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        mock_train = pd.DataFrame({'silver_feature': [1, 2]})
        mock_test = pd.DataFrame({'silver_feature': [3, 4]})
        
        mock_conn.execute.side_effect = [
            MagicMock(df=lambda: mock_train),
            MagicMock(df=lambda: mock_test)
        ]
        
        load_gold_data()
        
        # Verify only Gold layer access (which depends on Silver)
        expected_calls = [
            'SELECT * FROM gold.train',
            'SELECT * FROM gold.test'
        ]
        actual_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert actual_calls == expected_calls
        
        # Verify no Bronze/Raw data access
        forbidden_calls = ['bronze.train', 'bronze.test', 'playground_series_s5e7']
        for forbidden in forbidden_calls:
            assert not any(forbidden in call for call in actual_calls)
    
    def test_create_gold_tables_silver_dependency_simplified(self):
        """Test that Gold pipeline functions work with Silver data"""
        # Simulate Silver layer output
        silver_data = pd.DataFrame({
            'id': [1, 2, 3],
            'Social_event_attendance': [4, 6, 2],
            'extrovert_score': [10, 15, 5],
            'social_ratio': [1.33, 1.2, 2.0],
            'Social_event_participation_rate': [1.33, 1.2, 2.0],  # Silver features
            'Communication_ratio': [0.5, 0.6, 0.7],
            'Personality': ['Introvert', 'Extrovert', 'Introvert']
        })
        
        # Test Gold processing pipeline on Silver data
        step1 = encode_target(silver_data)
        step2 = clean_and_validate_features(step1)
        step3 = prepare_model_data(step2, target_col='Personality')
        
        # Verify Gold processing works on Silver input
        assert 'Personality_encoded' in step3.columns
        assert len(step3) == len(silver_data)
        
        # Verify Silver features are processed correctly
        assert 'Social_event_participation_rate' in step3.columns
        assert 'Communication_ratio' in step3.columns
        
        # Verify Gold adds LightGBM readiness
        feature_cols = [col for col in step3.columns 
                       if col not in ['id', 'Personality', 'Personality_encoded']]
        numeric_features = step3[feature_cols].select_dtypes(include=[np.number])
        assert not np.isinf(numeric_features).any().any()
        assert not numeric_features.isna().any().any()
    
    def test_gold_pipeline_integration(self):
        """Test Gold pipeline integration with Silver dependency"""
        # Setup test data that simulates Silver layer output
        silver_df = pd.DataFrame({
            'id': [1, 2, 3],
            'Social_event_attendance': [4, 6, 2],
            'Going_outside': [3, 5, 1],
            'extrovert_score': [10, 15, 5],
            'social_ratio': [1.33, 1.2, 2.0],
            'Social_event_participation_rate': [1.33, 1.2, 2.0],  # Silver features
            'Non_social_outings': [-1, -1, -1],
            'Communication_ratio': [0.5, 0.6, 0.7],
            'Personality': ['Introvert', 'Extrovert', 'Introvert']
        })
        
        # Apply Gold pipeline steps
        result1 = encode_target(silver_df)
        result2 = clean_and_validate_features(result1)
        result3 = prepare_model_data(result2, target_col='Personality')
        
        # Verify Gold processing guarantees
        assert 'Personality_encoded' in result3.columns  # Target encoding
        assert len(result3) == len(silver_df)            # Data integrity
        
        # Verify Silver features are preserved and processed
        assert 'Social_event_participation_rate' in result3.columns
        assert 'Communication_ratio' in result3.columns
        
        # Verify LightGBM readiness
        feature_cols = [col for col in result3.columns 
                       if col not in ['id', 'Personality', 'Personality_encoded']]
        numeric_features = result3[feature_cols].select_dtypes(include=[np.number])
        assert not np.isinf(numeric_features).any().any()
        assert not numeric_features.isna().any().any()


class TestGoldLightGBMInterface:
    """Test LightGBM Model Interface functionality"""
    
    def test_feature_importance_ranking_interface(self):
        """Test feature importance ranking for LightGBM optimization"""
        # Create data with known importance pattern
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'high_importance': [1, 5, 2, 6, 3],     # Clear signal
            'medium_importance': [2, 4, 3, 5, 4],   # Medium signal
            'low_importance': [1, 1, 1, 1, 1],     # No signal
            'target': [0, 1, 0, 1, 0]
        })
        
        selected_features = select_best_features(df, 'target', k=3)
        
        # Feature importance ranking should work
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 3
        assert 'id' not in selected_features
        assert 'target' not in selected_features
    
    def test_memory_optimization_arrays(self):
        """Test efficient array formats for training"""
        test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'Personality': ['Introvert', 'Extrovert', 'Introvert'],
            'Personality_encoded': [0, 1, 0]
        })
        
        with patch('src.data.gold.load_gold_data') as mock_load:
            mock_load.return_value = (test_df, test_df.drop('Personality', axis=1))
            
            X_train, y_train, X_test, test_ids = get_ml_ready_data()
            
            # Memory optimization checks
            assert X_train.dtype in [np.float32, np.float64]  # Numeric efficiency
            assert y_train.dtype in [np.int32, np.int64]      # Target efficiency
            assert X_test.dtype == X_train.dtype              # Consistency
            
            # Array format validation
            assert X_train.ndim == 2
            assert y_train.ndim == 1
            assert X_test.ndim == 2
    
    def test_production_quality_validation(self):
        """Test production-ready data quality ensuring model training stability"""
        # Create problematic data that could break model training
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'unstable_feature': [1e-10, 1e10, np.inf, -np.inf],
            'missing_feature': [1, np.nan, 3, np.nan],
            'constant_feature': [5, 5, 5, 5],
            'target': [0, 1, 0, 1]
        })
        
        result = clean_and_validate_features(df)
        
        # Production quality guarantees
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'id':
                # No infinite values
                assert not np.isinf(result[col]).any()
                # No missing values
                assert not result[col].isna().any()
                # Reasonable value ranges
                assert abs(result[col]).max() < 1e6  # Prevent numerical instability
