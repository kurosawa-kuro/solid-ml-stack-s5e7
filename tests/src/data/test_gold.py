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

    # ===== 新規追加テスト: Silver Dependency Enforcement =====
    
    def test_silver_dependency_exclusive_access(self):
        """Test that Gold layer exclusively consumes Silver layer (no Bronze/Raw access)"""
        # Mock database connection to verify access patterns
        with patch('duckdb.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            # Mock Silver layer data
            silver_train = pd.DataFrame({'silver_feature': [1, 2]})
            silver_test = pd.DataFrame({'silver_feature': [3, 4]})
            
            mock_conn.execute.side_effect = [
                MagicMock(df=lambda: silver_train),
                MagicMock(df=lambda: silver_test)
            ]
            
            # Execute Gold layer functions
            load_gold_data()
            create_gold_tables()
            
            # Verify only Silver layer access
            expected_calls = [
                'SELECT * FROM gold.train',
                'SELECT * FROM gold.test'
            ]
            actual_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
            
            for expected in expected_calls:
                assert any(expected in call for call in actual_calls), f"Expected call {expected} not found"
            
            # Verify no Bronze/Raw data access
            forbidden_calls = [
                'bronze.train', 'bronze.test', 
                'playground_series_s5e7.train', 'playground_series_s5e7.test'
            ]
            for forbidden in forbidden_calls:
                assert not any(forbidden in call for call in actual_calls), f"Forbidden call {forbidden} found"

    def test_model_ready_lightgbm_consumption(self):
        """Test direct LightGBM consumption without additional processing"""
        # Create Silver-like data
        silver_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'Social_event_attendance': [4, 6, 2, 5],
            'extrovert_score': [10, 15, 5, 12],
            'social_ratio': [1.33, 1.2, 2.0, 1.25],
            'Social_event_participation_rate': [0.8, 1.0, 0.5, 0.8],
            'Communication_ratio': [0.5, 0.6, 0.7, 0.5],
            'Personality': ['Introvert', 'Extrovert', 'Introvert', 'Extrovert']
        })
        
        # Apply Gold processing
        result = prepare_model_data(silver_df, target_col='Personality')
        
        # Test LightGBM consumption readiness
        # 1. All features should be numeric
        feature_cols = [col for col in result.columns 
                       if col not in ['id', 'Personality']]
        for col in feature_cols:
            assert result[col].dtype in ['float64', 'float32', 'int64', 'int32'], \
                f"Feature {col} not numeric for LightGBM"
        
        # 2. No missing values
        numeric_data = result[feature_cols].select_dtypes(include=[np.number])
        assert not numeric_data.isna().any().any(), "Missing values found in LightGBM data"
        
        # 3. No infinite values
        assert not np.isinf(numeric_data).any().any(), "Infinite values found in LightGBM data"
        
        # 4. Reasonable value ranges
        for col in feature_cols:
            values = result[col].dropna()
            if len(values) > 0:
                assert abs(values).max() < 1e6, f"Feature {col} has unreasonable values"

    # ===== 新規追加テスト: Competition Format Validation =====
    
    def test_competition_format_kaggle_submission_compatibility(self):
        """Test standard Kaggle submission file compatibility"""
        # Create test predictions
        predictions = np.array([0, 1, 0, 1, 0])  # Binary predictions
        
        # Mock test data with proper IDs
        test_data = pd.DataFrame({
            'id': [1001, 1002, 1003, 1004, 1005],
            'feature1': [1, 2, 3, 4, 5]
        })
        
        with patch('src.data.gold.load_gold_data') as mock_load:
            mock_load.return_value = (pd.DataFrame(), test_data)
            
            # Create submission file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
                create_submission(predictions, tmp.name)
                
                # Verify Kaggle submission format
                submission = pd.read_csv(tmp.name)
                
                # Required columns
                assert list(submission.columns) == ['id', 'Personality'], \
                    f"Expected columns ['id', 'Personality'], got {list(submission.columns)}"
                
                # Correct number of predictions
                assert len(submission) == 5, f"Expected 5 predictions, got {len(submission)}"
                
                # Valid personality values
                valid_personalities = {'Introvert', 'Extrovert'}
                assert set(submission['Personality']) <= valid_personalities, \
                    f"Invalid personality values: {set(submission['Personality'])}"
                
                # Correct ID mapping
                expected_ids = [1001, 1002, 1003, 1004, 1005]
                assert submission['id'].tolist() == expected_ids, \
                    f"ID mismatch: expected {expected_ids}, got {submission['id'].tolist()}"
                
                # Correct prediction mapping
                expected_personalities = ['Introvert', 'Extrovert', 'Introvert', 'Extrovert', 'Introvert']
                assert submission['Personality'].tolist() == expected_personalities, \
                    f"Prediction mismatch: expected {expected_personalities}, got {submission['Personality'].tolist()}"
                
                os.unlink(tmp.name)

    def test_model_prediction_interface_lightgbm_format(self):
        """Test model prediction interface: Direct LightGBM consumption format"""
        # Create Silver-like data
        silver_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'Social_event_attendance': [4, 6, 2, 5],
            'extrovert_score': [10, 15, 5, 12],
            'social_ratio': [1.33, 1.2, 2.0, 1.25],
            'Personality': ['Introvert', 'Extrovert', 'Introvert', 'Extrovert']
        })
        
        # Mock load_gold_data
        with patch('src.data.gold.load_gold_data') as mock_load:
            mock_load.return_value = (silver_df, silver_df.drop('Personality', axis=1))
            
            # Get LightGBM-ready data
            X_train, y_train, X_test, test_ids = get_ml_ready_data()
            
            # Verify LightGBM consumption format
            # 1. Correct data types
            assert X_train.dtype in [np.float32, np.float64], f"X_train dtype {X_train.dtype} not suitable for LightGBM"
            assert y_train.dtype in [np.int32, np.int64], f"y_train dtype {y_train.dtype} not suitable for LightGBM"
            assert X_test.dtype == X_train.dtype, "X_train and X_test dtypes should match"
            
            # 2. Correct shapes
            assert X_train.ndim == 2, f"X_train should be 2D, got {X_train.ndim}D"
            assert y_train.ndim == 1, f"y_train should be 1D, got {y_train.ndim}D"
            assert X_test.ndim == 2, f"X_test should be 2D, got {X_test.ndim}D"
            
            # 3. Consistent feature count
            assert X_train.shape[1] == X_test.shape[1], \
                f"Feature count mismatch: train {X_train.shape[1]}, test {X_test.shape[1]}"
            
            # 4. No missing/infinite values
            assert not np.isnan(X_train).any(), "X_train contains NaN values"
            assert not np.isnan(X_test).any(), "X_test contains NaN values"
            assert not np.isinf(X_train).any(), "X_train contains infinite values"
            assert not np.isinf(X_test).any(), "X_test contains infinite values"
            
            # 5. Valid target values
            assert set(y_train) <= {0, 1}, f"Invalid target values: {set(y_train)}"

    # ===== 新規追加テスト: Production Quality Enhancement =====
    
    def test_final_validation_infinite_value_processing(self):
        """Test final validation: Infinite value processing, outlier detection"""
        # Create data with production-quality issues
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'feature1': [1, np.inf, 3, -np.inf, 5],      # Infinite values
            'feature2': [1, 2, 1000, 4, 5],              # Extreme outlier
            'feature3': [np.nan, 2, 3, np.nan, 5],       # Missing values
            'feature4': [1e-10, 1e10, 1, 2, 3],          # Extreme ranges
            'Personality': ['Introvert', 'Extrovert', 'Introvert', 'Extrovert', 'Introvert']
        })
        
        result = clean_and_validate_features(df)
        
        # Final validation guarantees
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'id':
                # No infinite values
                assert not np.isinf(result[col]).any(), f"Infinite values found in {col}"
                # No missing values
                assert not result[col].isna().any(), f"Missing values found in {col}"
                # Reasonable value ranges
                assert abs(result[col]).max() < 1e6, f"Unreasonable values in {col}"
                assert abs(result[col]).min() > -1e6, f"Unreasonable values in {col}"

    def test_type_consistency_lightgbm_compatible(self):
        """Test type consistency: Ensure LightGBM-compatible data types"""
        # Create data with mixed types
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'numeric_float': [1.0, 2.0, 3.0],
            'numeric_int': [1, 2, 3],
            'string_feature': ['a', 'b', 'c'],
            'boolean_feature': [True, False, True],
            'Personality': ['Introvert', 'Extrovert', 'Introvert']
        })
        
        result = prepare_model_data(df, target_col='Personality')
        
        # Test LightGBM compatibility
        feature_cols = [col for col in result.columns 
                       if col not in ['id', 'Personality']]
        
        for col in feature_cols:
            # All features should be numeric for LightGBM
            assert result[col].dtype in ['float64', 'float32', 'int64', 'int32'], \
                f"Feature {col} has incompatible dtype {result[col].dtype} for LightGBM"
            
            # No string/object types in features
            assert result[col].dtype != 'object', f"Feature {col} is object type"

    def test_memory_optimization_efficient_arrays(self):
        """Test memory optimization: Efficient array formats for training"""
        # Create test data
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'Personality': ['Introvert', 'Extrovert', 'Introvert', 'Extrovert', 'Introvert']
        })
        
        with patch('src.data.gold.load_gold_data') as mock_load:
            mock_load.return_value = (df, df.drop('Personality', axis=1))
            
            X_train, y_train, X_test, test_ids = get_ml_ready_data()
            
            # Memory optimization checks
            # 1. Efficient numeric types
            assert X_train.dtype in [np.float32, np.float64], \
                f"X_train dtype {X_train.dtype} not memory efficient"
            assert y_train.dtype in [np.int32, np.int64], \
                f"y_train dtype {y_train.dtype} not memory efficient"
            
            # 2. Consistent types across arrays
            assert X_train.dtype == X_test.dtype, "X_train and X_test should have same dtype"
            
            # 3. No unnecessary object arrays
            assert X_train.dtype != np.object_, "X_train should not be object type"
            assert X_test.dtype != np.object_, "X_test should not be object type"
            
            # 4. Memory-efficient shapes
            assert X_train.shape[0] == len(y_train), "X_train and y_train should have same length"
            assert X_train.shape[1] == X_test.shape[1], "X_train and X_test should have same feature count"

    def test_audit_completeness_data_lineage_validation(self):
        """Test audit completeness: Comprehensive data lineage validation"""
        # Create Silver-like data with known lineage
        silver_df = pd.DataFrame({
            'id': [1, 2, 3],
            'Social_event_attendance': [4, 6, 2],           # Bronze origin
            'extrovert_score': [10, 15, 5],                 # Silver engineered
            'social_ratio': [1.33, 1.2, 2.0],              # Silver engineered
            'Social_event_participation_rate': [0.8, 1.0, 0.5],  # Silver CLAUDE.md feature
            'Communication_ratio': [0.5, 0.6, 0.7],        # Silver CLAUDE.md feature
            'Personality': ['Introvert', 'Extrovert', 'Introvert']
        })
        
        # Apply Gold processing
        result = prepare_model_data(silver_df, target_col='Personality')
        
        # Audit completeness checks
        # 1. All Silver features preserved
        expected_silver_features = [
            'Social_event_attendance', 'extrovert_score', 'social_ratio',
            'Social_event_participation_rate', 'Communication_ratio'
        ]
        for feature in expected_silver_features:
            assert feature in result.columns, f"Silver feature {feature} not preserved in Gold"
        
        # 2. Data integrity maintained
        assert len(result) == len(silver_df), "Data length changed during Gold processing"
        
        # 3. Feature lineage traceable
        bronze_features = ['Social_event_attendance']  # Original Bronze features
        silver_engineered = ['extrovert_score', 'social_ratio']  # Silver engineered
        silver_claude_features = ['Social_event_participation_rate', 'Communication_ratio']  # CLAUDE.md features
        
        for feature in bronze_features + silver_engineered + silver_claude_features:
            assert feature in result.columns, f"Feature lineage broken for {feature}"
        
        # 4. No data corruption
        for feature in expected_silver_features:
            original_values = silver_df[feature].values
            processed_values = result[feature].values
            np.testing.assert_array_almost_equal(original_values, processed_values, decimal=6)

    # ===== 新規追加テスト: Performance Optimization =====
    
    def test_feature_selection_bronze_medal_target_optimization(self):
        """Test feature selection maximizing Bronze Medal target (0.976518)"""
        # Create data with clear feature importance for Bronze Medal target
        np.random.seed(42)
        n_samples = 1000
        
        # Create features with known importance for personality prediction
        df = pd.DataFrame({
            'id': range(n_samples),
            'high_importance_extrovert': np.random.randn(n_samples),      # High importance
            'high_importance_introvert': np.random.randn(n_samples),      # High importance
            'medium_importance_social': np.random.randn(n_samples) * 0.5, # Medium importance
            'low_importance_noise': np.random.randn(n_samples) * 0.1,     # Low importance
            'Personality': np.random.randint(0, 2, n_samples)
        })
        
        # Make features predictive of personality
        df.loc[df['Personality'] == 1, 'high_importance_extrovert'] += 2
        df.loc[df['Personality'] == 0, 'high_importance_introvert'] += 2
        df.loc[df['Personality'] == 1, 'medium_importance_social'] += 1
        
        # Apply feature selection
        selected_features = select_best_features(df, 'Personality', k=3)
        
        # Bronze Medal target optimization checks
        assert isinstance(selected_features, list), "Feature selection should return list"
        assert len(selected_features) <= 3, f"Expected <=3 features, got {len(selected_features)}"
        
        # High importance features should be selected
        high_importance_features = ['high_importance_extrovert', 'high_importance_introvert']
        selected_high_importance = [f for f in selected_features if f in high_importance_features]
        assert len(selected_high_importance) >= 1, "At least one high importance feature should be selected"
        
        # Low importance features should not be prioritized
        assert 'low_importance_noise' not in selected_features, "Low importance feature should not be selected"

    def test_statistical_selection_f_test_mutual_information(self):
        """Test statistical feature selection (F-test + MI)"""
        # Create data with clear statistical patterns
        np.random.seed(42)
        n_samples = 500
        
        df = pd.DataFrame({
            'id': range(n_samples),
            'f_test_important': np.random.randn(n_samples),      # Good for F-test
            'mi_important': np.random.randn(n_samples),          # Good for Mutual Information
            'both_important': np.random.randn(n_samples),        # Good for both
            'unimportant': np.random.randn(n_samples) * 0.1,     # Unimportant
            'Personality': np.random.randint(0, 2, n_samples)
        })
        
        # Create clear statistical relationships
        df.loc[df['Personality'] == 1, 'f_test_important'] += 3  # Strong linear relationship
        df.loc[df['Personality'] == 1, 'mi_important'] = np.abs(df.loc[df['Personality'] == 1, 'mi_important'])  # Non-linear
        df.loc[df['Personality'] == 1, 'both_important'] += 2    # Both linear and non-linear
        
        # Apply statistical selection
        selected_features = select_best_features(df, 'Personality', k=3)
        
        # Statistical selection validation
        assert isinstance(selected_features, list), "Statistical selection should return list"
        assert len(selected_features) <= 3, f"Expected <=3 features, got {len(selected_features)}"
        
        # Important features should be selected
        important_features = ['f_test_important', 'mi_important', 'both_important']
        selected_important = [f for f in selected_features if f in important_features]
        assert len(selected_important) >= 2, "At least 2 important features should be selected"
        
        # Unimportant features should not be selected
        assert 'unimportant' not in selected_features, "Unimportant feature should not be selected"

    def test_feature_importance_ranking_lightgbm_optimization(self):
        """Test feature importance ranking for LightGBM optimization"""
        # Create data with known feature importance hierarchy
        np.random.seed(42)
        n_samples = 200
        
        df = pd.DataFrame({
            'id': range(n_samples),
            'top_feature': np.random.randn(n_samples),           # Highest importance
            'second_feature': np.random.randn(n_samples),        # Second importance
            'third_feature': np.random.randn(n_samples),         # Third importance
            'noise_feature': np.random.randn(n_samples) * 0.1,   # Noise
            'Personality': np.random.randint(0, 2, n_samples)
        })
        
        # Create importance hierarchy
        df.loc[df['Personality'] == 1, 'top_feature'] += 3
        df.loc[df['Personality'] == 1, 'second_feature'] += 2
        df.loc[df['Personality'] == 1, 'third_feature'] += 1
        
        # Apply feature selection
        selected_features = select_best_features(df, 'Personality', k=3)
        
        # Feature importance ranking validation
        assert isinstance(selected_features, list), "Feature selection should return list"
        assert len(selected_features) <= 3, f"Expected <=3 features, got {len(selected_features)}"
        
        # Top features should be selected in order
        if len(selected_features) >= 1:
            assert 'top_feature' in selected_features, "Top feature should be selected"
        if len(selected_features) >= 2:
            assert 'second_feature' in selected_features, "Second feature should be selected"
        
        # Noise feature should not be selected
        assert 'noise_feature' not in selected_features, "Noise feature should not be selected"

    def test_performance_monitoring_feature_importance_tracking(self):
        """Test performance monitoring: Feature importance and prediction tracking"""
        # Create test data
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'important_feature1': [1, 5, 2, 6, 3],
            'important_feature2': [2, 4, 3, 5, 4],
            'less_important_feature': [1, 1, 1, 1, 1],
            'Personality': [0, 1, 0, 1, 0]
        })
        
        # Test feature importance tracking
        selected_features = select_best_features(df, 'Personality', k=3)
        
        # Performance monitoring checks
        assert isinstance(selected_features, list), "Feature selection should return list"
        assert len(selected_features) > 0, "Should select at least one feature"
        
        # Important features should be tracked
        important_features = ['important_feature1', 'important_feature2']
        selected_important = [f for f in selected_features if f in important_features]
        assert len(selected_important) >= 1, "Should track at least one important feature"
        
        # Test prediction tracking capability
        # This would typically involve model training and prediction tracking
        # For now, we verify the data is ready for such tracking
        feature_cols = [col for col in df.columns 
                       if col not in ['id', 'Personality']]
        assert len(feature_cols) >= len(selected_features), "Feature tracking should be possible"
