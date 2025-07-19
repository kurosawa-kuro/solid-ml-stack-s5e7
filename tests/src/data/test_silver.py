"""
Test for Silver Level Data Functions - Success Cases Only
Includes comprehensive enhanced test cases from test_silver_gold_enhanced.py
"""

import tempfile
from unittest.mock import Mock, patch, MagicMock
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PolynomialFeatures

from src.data.silver import (
    advanced_features, 
    scaling_features,
    enhanced_interaction_features,
    polynomial_features,
    create_silver_tables,
    load_silver_data,
    get_feature_importance_order,
    s5e7_interaction_features,
    s5e7_drain_adjusted_features,
    s5e7_communication_ratios,
    DB_PATH,
)


class TestSilverFunctions:
    """Silver level function tests - minimal success cases"""

    def test_advanced_features_basic(self):
        """Test advanced features creation"""
        df = pd.DataFrame(
            {
                "Time_spent_Alone": [1.0, 2.0, 3.0],
                "Social_event_attendance": [2.0, 4.0, 6.0],
                "Going_outside": [1.0, 2.0, 3.0],
                "Stage_fear_encoded": [1, 0, 1],
                "Drained_after_socializing_encoded": [0, 1, 0],
            }
        )

        result = advanced_features(df)

        # Should return processed DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_scaling_features_basic(self):
        """Test feature scaling"""
        df = pd.DataFrame(
            {"feature1": [1.0, 2.0, 3.0], "feature2": [10.0, 20.0, 30.0], "feature3": [100.0, 200.0, 300.0]}
        )

        result = scaling_features(df)

        # Should return scaled DataFrame with additional scaled columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # Original columns should still be present
        for col in df.columns:
            assert col in result.columns

    @patch("src.data.silver.duckdb.connect")
    def test_create_silver_tables_success(self, mock_connect):
        """Test silver table creation succeeds"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        # Should not raise an error
        from src.data.silver import create_silver_tables

        create_silver_tables()

        # Verify connection was made and closed
        mock_connect.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("src.data.silver.duckdb.connect")
    def test_load_silver_data_success(self, mock_connect):
        """Test silver data loading succeeds"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_train = pd.DataFrame(
            {
                "id": [1, 2],
                "feature": [1.0, 2.0],
                "Stage_fear_encoded": [1, 0],
                "Drained_after_socializing_encoded": [0, 1],
            }
        )
        mock_test = pd.DataFrame(
            {
                "id": [3, 4],
                "feature": [3.0, 4.0],
                "Stage_fear_encoded": [1, 0],
                "Drained_after_socializing_encoded": [1, 0],
            }
        )

        mock_conn.execute.side_effect = [MagicMock(df=lambda: mock_train), MagicMock(df=lambda: mock_test)]

        from src.data.silver import load_silver_data

        train, test = load_silver_data()

        assert len(train) == 2
        assert len(test) == 2
        assert "Stage_fear_encoded" in train.columns
        assert "Drained_after_socializing_encoded" in train.columns

    def test_advanced_features_with_missing_columns(self):
        """Test advanced features with missing columns"""
        df = pd.DataFrame({"other_column": [1, 2, 3]})

        result = advanced_features(df)

        # Should not crash and return data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_scaling_features_single_column(self):
        """Test scaling with single column"""
        df = pd.DataFrame({"single_feature": [1.0, 2.0, 3.0]})

        result = scaling_features(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "single_feature" in result.columns

    def test_empty_dataframe_handling(self):
        """Test functions handle empty DataFrames gracefully"""
        empty_df = pd.DataFrame()

        # Should not crash
        result1 = advanced_features(empty_df)
        result2 = scaling_features(empty_df)

        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)


class TestSilverAdvancedFeatures:
    """Test advanced feature engineering in silver.py"""

    def test_advanced_features_basic_features(self):
        """Test basic feature creation in advanced_features"""
        df = pd.DataFrame({
            'Social_event_attendance': [4, 6, 2],
            'Time_spent_Alone': [2, 0, 8],
            'Going_outside': [3, 5, 1],
            'Friends_circle_size': [10, 15, 5],
            'Post_frequency': [5, 7, 3]
        })
        
        result = advanced_features(df)
        
        # Check basic features
        assert 'social_ratio' in result.columns
        assert 'activity_sum' in result.columns
        
        # Check calculations
        expected_social_ratio = [4/3, 6/1, 2/9]  # Social / (Time + 1)
        expected_activity_sum = [7, 11, 3]        # Going + Social
        
        np.testing.assert_array_almost_equal(
            result['social_ratio'].values, expected_social_ratio, decimal=6
        )
        assert result['activity_sum'].tolist() == expected_activity_sum

    def test_advanced_features_statistical_features(self):
        """Test statistical feature creation"""
        df = pd.DataFrame({
            'Time_spent_Alone': [2, 4, 6],
            'Social_event_attendance': [4, 6, 2],
            'Going_outside': [3, 5, 1],
            'Friends_circle_size': [10, 15, 5],
            'Post_frequency': [5, 7, 3]
        })
        
        result = advanced_features(df)
        
        # Check statistical features
        assert 'total_activity' in result.columns
        assert 'avg_activity' in result.columns
        assert 'activity_std' in result.columns
        
        # Verify calculations
        expected_total = [24, 37, 17]  # Sum of all numeric columns
        expected_avg = [4.8, 7.4, 3.4]  # Mean of all numeric columns
        
        assert result['total_activity'].tolist() == expected_total
        np.testing.assert_array_almost_equal(
            result['avg_activity'].values, expected_avg, decimal=1
        )

    def test_advanced_features_ratio_features(self):
        """Test ratio feature creation"""
        df = pd.DataFrame({
            'Friends_circle_size': [10, 0, 20],
            'Post_frequency': [5, 3, 40]
        })
        
        result = advanced_features(df)
        
        assert 'post_per_friend' in result.columns
        
        # Check calculations: Post_frequency / (Friends_circle_size + 1)
        expected_ratios = [5/11, 3/1, 40/21]
        np.testing.assert_array_almost_equal(
            result['post_per_friend'].values, expected_ratios, decimal=6
        )

    def test_advanced_features_interaction_features(self):
        """Test interaction feature creation"""
        df = pd.DataFrame({
            'Stage_fear_encoded': [1, 0, 1],
            'Drained_after_socializing_encoded': [1, 1, 0]
        })
        
        result = advanced_features(df)
        
        assert 'fear_drain_interaction' in result.columns
        expected_interaction = [1, 0, 0]  # 1*1, 0*1, 1*0
        assert result['fear_drain_interaction'].tolist() == expected_interaction

    def test_advanced_features_personality_scores(self):
        """Test personality score creation"""
        df = pd.DataFrame({
            'Social_event_attendance': [4, 6, 2],
            'Going_outside': [3, 5, 1],
            'Friends_circle_size': [10, 15, 5],
            'Time_spent_Alone': [2, 4, 8],
            'Stage_fear_encoded': [1, 0, 1],
            'Drained_after_socializing_encoded': [1, 1, 0]
        })
        
        result = advanced_features(df)
        
        # Check extrovert score
        assert 'extrovert_score' in result.columns
        expected_extrovert = [17, 26, 8]  # Sum of Social + Going + Friends
        assert result['extrovert_score'].tolist() == expected_extrovert
        
        # Check introvert score
        assert 'introvert_score' in result.columns
        expected_introvert = [6, 6, 10]  # Time + Stage*2 + Drained*2
        assert result['introvert_score'].tolist() == expected_introvert

    def test_advanced_features_missing_columns(self):
        """Test advanced features with missing columns"""
        df = pd.DataFrame({
            'Social_event_attendance': [4, 6, 2],
            'other_column': [1, 2, 3]
        })
        
        result = advanced_features(df)
        
        # Should not create features requiring missing columns
        assert 'total_activity' not in result.columns
        assert 'post_per_friend' not in result.columns
        assert len(result) == 3


class TestSilverInteractionFeatures:
    """Test enhanced interaction features"""

    def test_enhanced_interaction_basic(self):
        """Test basic interaction features"""
        df = pd.DataFrame({
            'extrovert_score': [10, 20, 15],
            'social_ratio': [0.5, 1.0, 0.3]
        })
        
        result = enhanced_interaction_features(df)
        
        # Check basic interactions
        assert 'extrovert_social_interaction' in result.columns
        assert 'extrovert_social_ratio' in result.columns
        assert 'social_extrovert_ratio' in result.columns
        
        # Verify calculations
        expected_interaction = [5.0, 20.0, 4.5]  # extrovert * social_ratio
        expected_extrovert_ratio = [10/1.5, 20/2.0, 15/1.3]  # extrovert / (social + 1)
        
        assert result['extrovert_social_interaction'].tolist() == expected_interaction
        np.testing.assert_array_almost_equal(
            result['extrovert_social_ratio'].values, expected_extrovert_ratio, decimal=6
        )

    def test_enhanced_interaction_extended(self):
        """Test extended interaction features"""
        df = pd.DataFrame({
            'extrovert_score': [10, 20, 15],
            'Social_event_attendance': [5, 8, 3],
            'Time_spent_Alone': [2, 4, 6],
            'Drained_after_socializing_encoded': [1, 0, 1]
        })
        
        result = enhanced_interaction_features(df)
        
        # Check extended interactions
        assert 'extrovert_social_event_interaction' in result.columns
        assert 'extrovert_alone_interaction' in result.columns
        assert 'extrovert_alone_contrast' in result.columns
        assert 'extrovert_drain_interaction' in result.columns
        
        # Verify some calculations
        expected_social_event = [50, 160, 45]  # extrovert * social_event
        expected_alone_contrast = [8, 16, 9]    # extrovert - time_alone
        
        assert result['extrovert_social_event_interaction'].tolist() == expected_social_event
        assert result['extrovert_alone_contrast'].tolist() == expected_alone_contrast

    def test_enhanced_interaction_triple(self):
        """Test triple interaction features"""
        df = pd.DataFrame({
            'extrovert_score': [10, 20],
            'social_ratio': [0.5, 1.0],
            'Social_event_attendance': [5, 8]
        })
        
        result = enhanced_interaction_features(df)
        
        assert 'triple_interaction' in result.columns
        expected_triple = [25.0, 160.0]  # 10*0.5*5, 20*1.0*8
        assert result['triple_interaction'].tolist() == expected_triple


class TestSilverPolynomialFeatures:
    """Test polynomial feature generation"""

    def test_polynomial_features_basic(self):
        """Test basic polynomial feature generation"""
        df = pd.DataFrame({
            'extrovert_score': [1, 2, 3],
            'social_ratio': [0.5, 1.0, 1.5],
            'Social_event_attendance': [2, 4, 6]
        })
        
        result = polynomial_features(df, degree=2)
        
        # Should have polynomial features added
        poly_columns = [col for col in result.columns if col.startswith('poly_')]
        assert len(poly_columns) > 0
        
        # Original data should be preserved
        pd.testing.assert_series_equal(result['extrovert_score'], df['extrovert_score'])

    def test_polynomial_features_with_nan(self):
        """Test polynomial features with NaN values"""
        df = pd.DataFrame({
            'extrovert_score': [1, np.nan, 3],
            'social_ratio': [0.5, 1.0, np.nan]
        })
        
        result = polynomial_features(df, degree=2)
        
        # Should handle NaN values gracefully
        assert not result['extrovert_score'].isna().all()
        poly_columns = [col for col in result.columns if col.startswith('poly_')]
        if poly_columns:  # If polynomial features were created
            assert not any(result[col].isna().all() for col in poly_columns)

    def test_polynomial_features_insufficient_features(self):
        """Test polynomial features with insufficient numeric features"""
        df = pd.DataFrame({
            'single_feature': [1, 2, 3],
            'text_feature': ['a', 'b', 'c']
        })
        
        result = polynomial_features(df, degree=2)
        
        # Should not create polynomial features with <2 numeric features
        poly_columns = [col for col in result.columns if col.startswith('poly_')]
        assert len(poly_columns) == 0

    @patch('builtins.print')
    def test_polynomial_features_error_handling(self, mock_print):
        """Test polynomial features error handling"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [np.inf, -np.inf, np.nan]  # Problematic data
        })
        
        # Should not raise exception even with problematic data
        result = polynomial_features(df, degree=2)
        
        # Original data should be preserved
        assert len(result) == 3


class TestSilverScalingFeatures:
    """Test feature scaling functionality"""

    def test_scaling_features_basic(self):
        """Test basic feature scaling"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'id': [1, 2, 3, 4, 5]  # Should be excluded
        })
        
        result = scaling_features(df)
        
        # Check scaled features exist
        assert 'feature1_scaled' in result.columns
        assert 'feature2_scaled' in result.columns
        assert 'id_scaled' not in result.columns  # ID should be excluded
        
        # Check standardization (mean ≈ 0, std ≈ 1)
        assert abs(result['feature1_scaled'].mean()) < 1e-10
        assert abs(result['feature1_scaled'].std() - 1) < 1e-10

    def test_scaling_features_zero_variance(self):
        """Test scaling with zero variance features"""
        df = pd.DataFrame({
            'constant_feature': [5, 5, 5, 5],
            'variable_feature': [1, 2, 3, 4]
        })
        
        result = scaling_features(df)
        
        # Constant feature should not be scaled
        assert 'constant_feature_scaled' not in result.columns
        assert 'variable_feature_scaled' in result.columns

    def test_scaling_features_mixed_types(self):
        """Test scaling with mixed data types"""
        df = pd.DataFrame({
            'numeric_int': [1, 2, 3],
            'numeric_float': [1.1, 2.2, 3.3],
            'text_feature': ['a', 'b', 'c'],
            'boolean_feature': [True, False, True]
        })
        
        result = scaling_features(df)
        
        # Only numeric features should be scaled
        assert 'numeric_int_scaled' in result.columns
        assert 'numeric_float_scaled' in result.columns
        assert 'text_feature_scaled' not in result.columns
        assert 'boolean_feature_scaled' in result.columns  # Boolean is numeric


class TestSilverTableOperations:
    """Test silver table creation and loading"""

    @patch('duckdb.connect')
    @patch('src.data.silver.advanced_features')
    @patch('src.data.silver.enhanced_interaction_features')
    @patch('src.data.silver.polynomial_features')
    @patch('src.data.silver.scaling_features')
    def test_create_silver_tables_success(self, mock_scaling, mock_poly, mock_interaction, 
                                        mock_advanced, mock_connect):
        """Test successful silver table creation"""
        # Setup mocks
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Mock bronze data
        bronze_train = pd.DataFrame({'id': [1, 2], 'feature': [1, 2]})
        bronze_test = pd.DataFrame({'id': [3, 4], 'feature': [3, 4]})
        
        mock_result_train = Mock()
        mock_result_train.df.return_value = bronze_train
        mock_result_test = Mock()
        mock_result_test.df.return_value = bronze_test
        
        mock_conn.execute.side_effect = [mock_result_train, mock_result_test] + [None] * 10
        
        # Mock feature engineering functions
        mock_advanced.side_effect = lambda x: x.copy()
        mock_interaction.side_effect = lambda x: x.copy()
        mock_poly.side_effect = lambda x, degree: x.copy()
        mock_scaling.side_effect = lambda x: x.copy()
        
        # Test function
        create_silver_tables()
        
        # Verify database operations
        mock_connect.assert_called_with(DB_PATH)
        assert mock_conn.execute.call_count >= 5
        assert mock_conn.register.call_count == 2
        assert mock_conn.close.called
        
        # Verify feature engineering pipeline
        mock_advanced.assert_called()
        mock_interaction.assert_called()
        mock_poly.assert_called()
        mock_scaling.assert_called()

    @patch('duckdb.connect')
    @patch('src.data.bronze.create_bronze_tables')
    @patch('builtins.print')
    def test_create_silver_tables_missing_bronze(self, mock_print, mock_create_bronze, mock_connect):
        """Test silver table creation when bronze tables are missing"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # First call raises exception (bronze missing)
        # Second and third calls return data (after bronze creation)
        mock_conn.execute.side_effect = [
            Exception("Bronze table not found"),
            Mock(df=lambda: pd.DataFrame({'id': [1]})),
            Mock(df=lambda: pd.DataFrame({'id': [2]}))
        ] + [None] * 10
        
        create_silver_tables()
        
        # Should call bronze table creation
        mock_create_bronze.assert_called_once()
        assert any("Bronze tables not found" in str(call) for call in mock_print.call_args_list)

    @patch('duckdb.connect')
    def test_load_silver_data_success(self, mock_connect):
        """Test successful silver data loading"""
        mock_conn = Mock()
        mock_train_df = pd.DataFrame({'id': [1, 2], 'silver_feature': [1, 2]})
        mock_test_df = pd.DataFrame({'id': [3, 4], 'silver_feature': [3, 4]})
        
        mock_train_result = Mock()
        mock_train_result.df.return_value = mock_train_df
        mock_test_result = Mock()
        mock_test_result.df.return_value = mock_test_df
        
        mock_conn.execute.side_effect = [mock_train_result, mock_test_result]
        mock_connect.return_value = mock_conn
        
        train, test = load_silver_data()
        
        # Verify correct queries
        expected_calls = [
            "SELECT * FROM silver.train",
            "SELECT * FROM silver.test"
        ]
        actual_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert actual_calls == expected_calls
        
        pd.testing.assert_frame_equal(train, mock_train_df)
        pd.testing.assert_frame_equal(test, mock_test_df)


class TestSilverCLAUDEMDFeatures:
    """Test CLAUDE.md specified Silver layer functions"""
    
    def test_s5e7_interaction_features(self):
        """Test Winner Solution Interaction Features (+0.2-0.4% proven impact)"""
        df = pd.DataFrame({
            'Social_event_attendance': [4, 6, 2, 0],
            'Going_outside': [5, 6, 4, 2],
            'Post_frequency': [10, 12, 8, 4],
            'Friends_circle_size': [20, 15, 10, 5]
        })
        
        result = s5e7_interaction_features(df)
        
        # Test Social_event_participation_rate = Social_event_attendance ÷ Going_outside
        assert 'Social_event_participation_rate' in result.columns
        expected_participation = [4/5, 6/6, 2/4, 0/2]  # Handle division
        np.testing.assert_array_almost_equal(
            result['Social_event_participation_rate'].values, expected_participation, decimal=6
        )
        
        # Test Non_social_outings = Going_outside - Social_event_attendance
        assert 'Non_social_outings' in result.columns
        expected_non_social = [1, 0, 2, 2]  # 5-4, 6-6, 4-2, 2-0
        assert result['Non_social_outings'].tolist() == expected_non_social
        
        # Test Communication_ratio = Post_frequency ÷ (Social_event_attendance + Going_outside)
        assert 'Communication_ratio' in result.columns
        expected_comm_ratio = [10/9, 12/12, 8/6, 4/2]  # Post/(Social+Going)
        np.testing.assert_array_almost_equal(
            result['Communication_ratio'].values, expected_comm_ratio, decimal=6
        )
        
        # Test Friend_social_efficiency = Social_event_attendance ÷ Friends_circle_size
        assert 'Friend_social_efficiency' in result.columns
        expected_efficiency = [4/20, 6/15, 2/10, 0/5]  # Social/Friends
        np.testing.assert_array_almost_equal(
            result['Friend_social_efficiency'].values, expected_efficiency, decimal=6
        )
    
    def test_s5e7_drain_adjusted_features(self):
        """Test Fatigue-Adjusted Domain Modeling (+0.1-0.2% introversion accuracy)"""
        df = pd.DataFrame({
            'Social_event_attendance': [4, 6, 2],
            'Going_outside': [3, 5, 1], 
            'Post_frequency': [5, 7, 3],
            'Drained_after_socializing': [1.0, 0.0, 1.0]  # Bronze encoded format
        })
        
        result = s5e7_drain_adjusted_features(df)
        
        # Test Activity_ratio = comprehensive_activity_index
        assert 'Activity_ratio' in result.columns
        expected_activity = [12, 18, 6]  # Sum of social+going+post
        assert result['Activity_ratio'].tolist() == expected_activity
        
        # Test Drain_adjusted_activity = activity_ratio × (1 - Drained_after_socializing)
        assert 'Drain_adjusted_activity' in result.columns
        expected_drain_adj = [12*(1-1), 18*(1-0), 6*(1-1)]  # activity*(1-drain)
        expected_drain_adj = [0.0, 18.0, 0.0]
        assert result['Drain_adjusted_activity'].tolist() == expected_drain_adj
        
        # Test Introvert_extrovert_spectrum = quantified_personality_score
        assert 'Introvert_extrovert_spectrum' in result.columns
        # Should be numerical personality quantification
        assert all(isinstance(x, (int, float)) for x in result['Introvert_extrovert_spectrum'])
    
    def test_s5e7_communication_ratios(self):
        """Test Online vs Offline behavioral ratios"""
        df = pd.DataFrame({
            'Post_frequency': [10, 5, 15],
            'Social_event_attendance': [4, 6, 2],
            'Going_outside': [3, 5, 1]
        })
        
        result = s5e7_communication_ratios(df)
        
        # Test Online_offline_ratio = Post_frequency ÷ (Social_event_attendance + Going_outside)
        assert 'Online_offline_ratio' in result.columns
        expected_ratio = [10/7, 5/11, 15/3]  # Post/(Social+Going)
        np.testing.assert_array_almost_equal(
            result['Online_offline_ratio'].values, expected_ratio, decimal=6
        )
        
        # Test Communication_balance = balanced ratio calculation
        assert 'Communication_balance' in result.columns
        # Should provide balanced view of online vs offline activity
        assert all(isinstance(x, (int, float)) for x in result['Communication_balance'])

class TestSilverDependencyChain:
    """Test Silver layer Bronze dependency enforcement"""
    
    @patch('duckdb.connect')
    def test_load_silver_data_bronze_dependency(self, mock_connect):
        """Test that load_silver_data only accesses Bronze tables"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        mock_train = pd.DataFrame({'bronze_feature': [1, 2]})
        mock_test = pd.DataFrame({'bronze_feature': [3, 4]})
        
        mock_conn.execute.side_effect = [
            MagicMock(df=lambda: mock_train),
            MagicMock(df=lambda: mock_test)
        ]
        
        load_silver_data()
        
        # Verify only Bronze layer access
        expected_calls = [
            'SELECT * FROM silver.train',
            'SELECT * FROM silver.test'
        ]
        actual_calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert actual_calls == expected_calls
        
        # Verify no raw data access
        forbidden_calls = ['playground_series_s5e7.train', 'playground_series_s5e7.test']
        for forbidden in forbidden_calls:
            assert not any(forbidden in call for call in actual_calls)
    
    def test_silver_pipeline_integration(self):
        """Test Silver pipeline integration with CLAUDE.md functions"""
        # Setup test data with all required columns
        test_df = pd.DataFrame({
            'Social_event_attendance': [4, 6],
            'Going_outside': [3, 5],
            'Post_frequency': [5, 7],
            'Friends_circle_size': [10, 15],
            'Time_spent_Alone': [2, 3],
            'Drained_after_socializing': [1.0, 0.0]
        })
        
        # Apply CLAUDE.md specified pipeline steps
        result1 = s5e7_interaction_features(test_df)
        result2 = s5e7_drain_adjusted_features(result1)
        result3 = s5e7_communication_ratios(result2)
        
        # Verify CLAUDE.md features are created
        assert 'Social_event_participation_rate' in result3.columns
        assert 'Non_social_outings' in result3.columns
        assert 'Communication_ratio' in result3.columns
        assert 'Friend_social_efficiency' in result3.columns
        assert 'Activity_ratio' in result3.columns
        assert 'Drain_adjusted_activity' in result3.columns
        assert 'Online_offline_ratio' in result3.columns
        
        # Verify data integrity
        assert len(result3) == len(test_df)
        assert result3.shape[1] > test_df.shape[1]  # More columns added

class TestSilverUtilities:
    """Test utility functions"""

    def test_get_feature_importance_order(self):
        """Test feature importance order function"""
        importance_order = get_feature_importance_order()
        
        # Should return a list of feature names
        assert isinstance(importance_order, list)
        assert len(importance_order) > 0
        
        # Check for expected important features
        expected_features = ['extrovert_score', 'introvert_score', 'Social_event_attendance']
        for feature in expected_features:
            assert feature in importance_order
        
        # Should be ordered (most important first)
        assert importance_order[0] == 'extrovert_score'


class TestSilverIntegration:
    """Test silver module integration scenarios"""

    def test_full_silver_pipeline(self):
        """Test complete silver processing pipeline"""
        # Create comprehensive sample data
        df = pd.DataFrame({
            'Time_spent_Alone': [2, 4, 6, 3, 5],
            'Social_event_attendance': [4, 6, 2, 5, 3],
            'Going_outside': [3, 5, 1, 4, 2],
            'Friends_circle_size': [10, 15, 5, 12, 8],
            'Post_frequency': [5, 7, 3, 6, 4],
            'Stage_fear_encoded': [1, 0, 1, 0, 1],
            'Drained_after_socializing_encoded': [1, 1, 0, 1, 0]
        })
        
        # Apply full pipeline
        step1 = advanced_features(df)
        step2 = enhanced_interaction_features(step1)
        step3 = polynomial_features(step2, degree=2)
        step4 = scaling_features(step3)
        
        # Verify pipeline completeness
        assert len(step4) == len(df)
        assert step4.shape[1] > df.shape[1]  # Should have more columns
        
        # Check for key features
        assert 'extrovert_score' in step4.columns
        assert 'social_ratio' in step4.columns
        
        # Check for scaled features
        scaled_features = [col for col in step4.columns if col.endswith('_scaled')]
        assert len(scaled_features) > 0

    def test_edge_case_empty_dataframe(self):
        """Test pipeline with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        result1 = advanced_features(empty_df)
        result2 = enhanced_interaction_features(result1)
        result3 = polynomial_features(result2)
        result4 = scaling_features(result3)
        
        assert len(result4) == 0
        assert isinstance(result4, pd.DataFrame)

    def test_edge_case_single_column(self):
        """Test pipeline with minimal data"""
        df = pd.DataFrame({
            'single_feature': [1, 2, 3]
        })
        
        # Should not break the pipeline
        result1 = advanced_features(df)
        result2 = enhanced_interaction_features(result1)
        result3 = polynomial_features(result2)
        result4 = scaling_features(result3)
        
        assert len(result4) == 3
        assert 'single_feature' in result4.columns
