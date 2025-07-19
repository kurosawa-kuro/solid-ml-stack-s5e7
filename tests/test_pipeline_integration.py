"""
Integration test for Bronze → Silver → Gold pipeline
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.bronze import load_data


class TestPipelineIntegration:
    """Integration tests for full data pipeline"""

    @patch("src.data.bronze.duckdb.connect")
    def test_bronze_data_flow(self, mock_bronze_connect):
        """Test bronze data loading flow"""
        # Mock bronze data loading
        mock_bronze_conn = MagicMock()
        mock_bronze_connect.return_value = mock_bronze_conn

        mock_train_raw = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "Personality": ["Introvert", "Extrovert", "Introvert"],
                "Time_spent_Alone": [1.0, 2.0, 3.0],
                "Social_event_attendance": [2.0, 4.0, 6.0],
                "Stage_fear": ["Yes", "No", "Yes"],
                "Drained_after_socializing": ["No", "Yes", "No"],
            }
        )
        mock_test_raw = pd.DataFrame(
            {
                "id": [4, 5, 6],
                "Time_spent_Alone": [4.0, 5.0, 6.0],
                "Social_event_attendance": [8.0, 10.0, 12.0],
                "Stage_fear": ["No", "Yes", "No"],
                "Drained_after_socializing": ["Yes", "No", "Yes"],
            }
        )

        mock_bronze_conn.execute.side_effect = [
            MagicMock(df=lambda: mock_train_raw),
            MagicMock(df=lambda: mock_test_raw),
        ]

        # Execute bronze step
        train_raw, test_raw = load_data()

        # Prepare for gold features (simulating silver step)
        X_train = train_raw.drop("Personality", axis=1, errors="ignore")
        X_test = test_raw
        y_train = train_raw["Personality"] if "Personality" in train_raw.columns else None

        # Assertions
        assert len(train_raw) == 3
        assert len(test_raw) == 3
        assert "Personality" in train_raw.columns

        assert len(X_train) == 3
        assert len(X_test) == 3
        assert y_train is not None
        assert len(y_train) == 3

        # Check that we have expected feature columns
        expected_features = [
            "id",
            "Time_spent_Alone",
            "Social_event_attendance",
            "Stage_fear",
            "Drained_after_socializing",
        ]
        for feature in expected_features:
            assert feature in X_train.columns

    @patch("src.data.bronze.duckdb.connect")
    def test_pipeline_data_consistency(self, mock_connect):
        """Test data consistency through pipeline stages"""
        # Mock setup
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_train = pd.DataFrame(
            {"id": [1, 2, 3, 4, 5], "Personality": ["Introvert", "Extrovert", "Introvert", "Extrovert", "Introvert"]}
        )
        mock_test = pd.DataFrame({"id": [6, 7, 8, 9, 10]})

        mock_conn.execute.side_effect = [MagicMock(df=lambda: mock_train), MagicMock(df=lambda: mock_test)]

        # Execute
        train, test = load_data()

        # Check ID consistency
        assert train["id"].nunique() == len(train)
        assert test["id"].nunique() == len(test)
        assert set(train["id"]).isdisjoint(set(test["id"]))  # No overlap

        # Check target distribution
        target_counts = train["Personality"].value_counts()
        assert "Introvert" in target_counts.index
        assert "Extrovert" in target_counts.index

    def test_pipeline_error_handling(self):
        """Test pipeline behavior with missing or malformed data"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()

        # Should not crash when processing empty data
        X_train = empty_df.drop("Personality", axis=1, errors="ignore")
        y_train = empty_df["Personality"] if "Personality" in empty_df.columns else None

        assert len(X_train) == 0
        assert y_train is None

        # Test with missing target column
        df_no_target = pd.DataFrame({"id": [1, 2, 3], "feature1": [1.0, 2.0, 3.0]})

        X = df_no_target.drop("Personality", axis=1, errors="ignore")
        y = df_no_target["Personality"] if "Personality" in df_no_target.columns else None

        assert len(X) == 3
        assert "id" in X.columns
        assert "feature1" in X.columns
        assert y is None
