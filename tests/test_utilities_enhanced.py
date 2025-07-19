"""
Comprehensive test coverage for src/util/ modules
Targeting high-impact functions to improve coverage from 19-24% to 80%+
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import pytest
import requests

from src.util.notifications import (
    WebhookNotifier,
)

from src.util.time_tracker import (
    WorkflowTimeTracker,
)


class TestWebhookNotifier:
    """Test WebhookNotifier functionality"""

    def test_init_with_url(self):
        """Test initialization with explicit webhook URL"""
        url = "https://discord.com/api/webhooks/test"
        notifier = WebhookNotifier(webhook_url=url)
        
        assert notifier.webhook_url == url

    @patch.dict(os.environ, {'WEBHOOK_DISCORD': 'https://discord.com/api/webhooks/env'})
    def test_init_with_env_var(self):
        """Test initialization with environment variable"""
        notifier = WebhookNotifier()
        
        assert notifier.webhook_url == "https://discord.com/api/webhooks/env"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_no_url(self):
        """Test initialization without URL raises error"""
        with pytest.raises(ValueError, match="Webhook URL not provided"):
            WebhookNotifier()

    @patch('requests.post')
    def test_send_message_success(self, mock_post):
        """Test successful message sending"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response
        
        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        result = notifier.send_message("Test message")
        
        assert result is True
        mock_post.assert_called_once()
        
        # Check payload structure
        call_args = mock_post.call_args
        assert call_args[1]['json']['content'] == "Test message"
        assert call_args[1]['json']['username'] == "ML Pipeline"

    @patch('requests.post')
    def test_send_message_custom_username(self, mock_post):
        """Test message sending with custom username"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response
        
        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        result = notifier.send_message("Test", username="Custom Bot")
        
        assert result is True
        call_args = mock_post.call_args
        assert call_args[1]['json']['username'] == "Custom Bot"

    @patch('requests.post')
    def test_send_message_with_embeds(self, mock_post):
        """Test message sending with embeds"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response
        
        embeds = [{"title": "Test Embed", "description": "Test description"}]
        
        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        result = notifier.send_message("Test", embeds=embeds)
        
        assert result is True
        call_args = mock_post.call_args
        assert call_args[1]['json']['embeds'] == embeds

    @patch('requests.post')
    def test_send_message_failure_status(self, mock_post):
        """Test message sending with failure status code"""
        mock_response = Mock()
        mock_response.status_code = 400  # Bad request
        mock_post.return_value = mock_response
        
        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        result = notifier.send_message("Test message")
        
        assert result is False

    @patch('requests.post')
    @patch('builtins.print')
    def test_send_message_exception(self, mock_print, mock_post):
        """Test message sending with exception handling"""
        mock_post.side_effect = requests.RequestException("Network error")
        
        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        result = notifier.send_message("Test message")
        
        assert result is False
        mock_print.assert_called_once()
        assert "Webhook notification failed" in mock_print.call_args[0][0]

    def test_send_message_no_webhook_url(self):
        """Test message sending when webhook URL is None"""
        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        notifier.webhook_url = None
        
        result = notifier.send_message("Test message")
        
        assert result is False


class TestWorkflowTimeTracker:
    """Test WorkflowTimeTracker functionality"""

    def test_init_new_file(self):
        """Test initialization with new JSON file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            assert tracker.db_path == Path(db_path)
            assert tracker.data == {"workflows": {}}

    def test_init_existing_file(self):
        """Test initialization with existing JSON file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            
            # Create existing data
            existing_data = {
                "workflows": {
                    "test_workflow": [
                        {"duration": 10.5, "timestamp": "2024-01-01T12:00:00"}
                    ]
                }
            }
            
            with open(db_path, 'w') as f:
                json.dump(existing_data, f)
            
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            assert tracker.data == existing_data

    def test_init_corrupted_file(self):
        """Test initialization with corrupted JSON file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            
            # Create corrupted JSON
            with open(db_path, 'w') as f:
                f.write("invalid json content")
            
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Should fallback to default structure
            assert tracker.data == {"workflows": {}}

    def test_start_workflow(self):
        """Test starting workflow tracking"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            start_time = tracker.start_workflow("test_workflow")
            
            assert isinstance(start_time, float)
            assert start_time > 0

    @patch('time.time')
    def test_end_workflow_success(self, mock_time):
        """Test successful workflow completion"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Mock time progression
            mock_time.side_effect = [1000.0, 1010.5]  # 10.5 second duration
            
            start_time = tracker.start_workflow("test_workflow")
            duration = tracker.end_workflow("test_workflow", start_time)
            
            assert duration == 10.5
            
            # Check data was saved
            assert "test_workflow" in tracker.data["workflows"]
            workflow_data = tracker.data["workflows"]["test_workflow"]
            assert len(workflow_data) == 1
            assert workflow_data[0]["duration"] == 10.5
            assert "timestamp" in workflow_data[0]

    def test_end_workflow_missing_start(self):
        """Test ending workflow without start time"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            with pytest.raises(ValueError, match="start_time must be provided"):
                tracker.end_workflow("test_workflow")

    def test_get_average_time_no_history(self):
        """Test getting average time with no history"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            avg_time = tracker.get_average_time("nonexistent_workflow")
            
            assert avg_time is None

    def test_get_average_time_with_history(self):
        """Test getting average time with history"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Add some history
            tracker.data["workflows"]["test_workflow"] = [
                {"duration": 10.0, "timestamp": "2024-01-01T12:00:00"},
                {"duration": 20.0, "timestamp": "2024-01-01T12:05:00"},
                {"duration": 30.0, "timestamp": "2024-01-01T12:10:00"}
            ]
            
            avg_time = tracker.get_average_time("test_workflow")
            
            assert avg_time == 20.0  # (10 + 20 + 30) / 3

    def test_get_average_time_with_limit(self):
        """Test getting average time with limited history"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Add multiple entries
            tracker.data["workflows"]["test_workflow"] = [
                {"duration": 10.0, "timestamp": "2024-01-01T12:00:00"},
                {"duration": 20.0, "timestamp": "2024-01-01T12:05:00"},
                {"duration": 30.0, "timestamp": "2024-01-01T12:10:00"},
                {"duration": 40.0, "timestamp": "2024-01-01T12:15:00"}
            ]
            
            # Get average of last 2 runs
            avg_time = tracker.get_average_time("test_workflow", last_n_runs=2)
            
            assert avg_time == 35.0  # (30 + 40) / 2

    def test_predict_completion_time_no_history(self):
        """Test predicting completion time with no history"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            completion_time = tracker.predict_completion_time("nonexistent_workflow")
            
            assert completion_time is None

    @patch('src.util.time_tracker.datetime')
    def test_predict_completion_time_with_history(self, mock_datetime):
        """Test predicting completion time with history"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Mock current time
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            # Add history (average 15 seconds)
            tracker.data["workflows"]["test_workflow"] = [
                {"duration": 10.0, "timestamp": "2024-01-01T11:00:00"},
                {"duration": 20.0, "timestamp": "2024-01-01T11:30:00"}
            ]
            
            completion_time = tracker.predict_completion_time("test_workflow")
            
            expected_time = mock_now + timedelta(seconds=15)
            assert completion_time == expected_time

    def test_get_workflow_stats_no_history(self):
        """Test getting workflow stats with no history"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            stats = tracker.get_workflow_stats("nonexistent_workflow")
            
            assert stats == {
                "count": 0,
                "average": None,
                "median": None,
                "min": None,
                "max": None,
                "total": 0
            }

    def test_get_workflow_stats_with_history(self):
        """Test getting workflow stats with history"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Add history
            tracker.data["workflows"]["test_workflow"] = [
                {"duration": 10.0, "timestamp": "2024-01-01T12:00:00"},
                {"duration": 20.0, "timestamp": "2024-01-01T12:05:00"},
                {"duration": 30.0, "timestamp": "2024-01-01T12:10:00"}
            ]
            
            stats = tracker.get_workflow_stats("test_workflow")
            
            assert stats["count"] == 3
            assert stats["average"] == 20.0
            assert stats["median"] == 20.0
            assert stats["min"] == 10.0
            assert stats["max"] == 30.0
            assert stats["total"] == 60.0

    def test_get_all_workflows(self):
        """Test getting all tracked workflows"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Add multiple workflows
            tracker.data["workflows"] = {
                "workflow1": [{"duration": 10.0, "timestamp": "2024-01-01T12:00:00"}],
                "workflow2": [{"duration": 20.0, "timestamp": "2024-01-01T12:05:00"}]
            }
            
            workflows = tracker.get_all_workflows()
            
            assert set(workflows) == {"workflow1", "workflow2"}

    def test_clear_workflow_history(self):
        """Test clearing workflow history"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Add history
            tracker.data["workflows"]["test_workflow"] = [
                {"duration": 10.0, "timestamp": "2024-01-01T12:00:00"}
            ]
            
            tracker.clear_workflow_history("test_workflow")
            
            assert "test_workflow" not in tracker.data["workflows"]

    def test_clear_workflow_history_nonexistent(self):
        """Test clearing history for nonexistent workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Should not raise exception
            tracker.clear_workflow_history("nonexistent_workflow")

    def test_save_data_creates_directory(self):
        """Test that save_data creates necessary directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "dir", "test_times.json")
            tracker = WorkflowTimeTracker(db_path=nested_path)
            
            # Trigger save by ending a workflow
            tracker.data["workflows"]["test"] = [{"duration": 1.0, "timestamp": "2024-01-01T12:00:00"}]
            tracker._save_data()
            
            assert os.path.exists(nested_path)

    def test_workflow_context_manager(self):
        """Test workflow tracking as context manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Test context manager functionality if implemented
            # This would require the context manager methods to be implemented
            # For now, we'll test the manual start/end pattern
            
            start_time = tracker.start_workflow("context_test")
            time.sleep(0.1)  # Small delay
            duration = tracker.end_workflow("context_test", start_time)
            
            assert duration > 0
            assert "context_test" in tracker.data["workflows"]


class TestUtilitiesIntegration:
    """Test integration scenarios for utility modules"""

    @patch('requests.post')
    def test_notification_time_tracking_integration(self, mock_post):
        """Test integration between notifications and time tracking"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup time tracker
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Setup notifier
            notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
            
            # Simulate ML workflow with notifications
            start_time = tracker.start_workflow("ml_training")
            time.sleep(0.1)
            duration = tracker.end_workflow("ml_training", start_time)
            
            # Send completion notification
            message = f"ML training completed in {duration:.2f} seconds"
            success = notifier.send_message(message)
            
            assert success
            assert duration > 0
            mock_post.assert_called_once()

    def test_error_resilience(self):
        """Test utility modules handle errors gracefully"""
        # Test time tracker with invalid paths
        invalid_tracker = WorkflowTimeTracker(db_path="/invalid/path/test.json")
        
        # Should still work for basic operations
        start_time = invalid_tracker.start_workflow("test")
        assert isinstance(start_time, float)
        
        # Test notifier with invalid URL
        invalid_notifier = WebhookNotifier("invalid-url")
        result = invalid_notifier.send_message("test")
        
        # Should return False but not crash
        assert result is False

    def test_concurrent_usage_simulation(self):
        """Test utilities can handle concurrent-like usage"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            
            # Simulate multiple trackers (like in multi-process scenario)
            tracker1 = WorkflowTimeTracker(db_path=db_path)
            tracker2 = WorkflowTimeTracker(db_path=db_path)
            
            # Both should start with same data
            assert tracker1.data == tracker2.data
            
            # Simulate workflow tracking
            start1 = tracker1.start_workflow("workflow1")
            start2 = tracker2.start_workflow("workflow2")
            
            time.sleep(0.1)
            
            duration1 = tracker1.end_workflow("workflow1", start1)
            duration2 = tracker2.end_workflow("workflow2", start2)
            
            assert duration1 > 0
            assert duration2 > 0

    @patch.dict(os.environ, {'WEBHOOK_DISCORD': 'https://test.webhook.url'})
    def test_environment_configuration(self):
        """Test utilities respect environment configuration"""
        # Test webhook notifier reads from environment
        notifier = WebhookNotifier()
        assert notifier.webhook_url == "https://test.webhook.url"
        
        # Test time tracker uses default path
        tracker = WorkflowTimeTracker()
        assert str(tracker.db_path) == "data/workflow_times.json"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios"""

    def test_webhook_notifier_edge_cases(self):
        """Test webhook notifier edge cases"""
        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        
        # Test with empty message
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 204
            mock_post.return_value = mock_response
            
            result = notifier.send_message("")
            assert result is True

        # Test with None embeds
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 204
            mock_post.return_value = mock_response
            
            result = notifier.send_message("test", embeds=None)
            assert result is True

    def test_time_tracker_edge_cases(self):
        """Test time tracker edge cases"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path=db_path)
            
            # Test with empty workflow name
            start_time = tracker.start_workflow("")
            duration = tracker.end_workflow("", start_time)
            assert duration >= 0
            
            # Test stats for workflow with single entry
            tracker.data["workflows"]["single"] = [
                {"duration": 10.0, "timestamp": "2024-01-01T12:00:00"}
            ]
            
            stats = tracker.get_workflow_stats("single")
            assert stats["count"] == 1
            assert stats["average"] == 10.0
            assert stats["median"] == 10.0
            assert stats["min"] == 10.0
            assert stats["max"] == 10.0

    def test_data_persistence(self):
        """Test data persistence across instances"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_times.json")
            
            # First instance
            tracker1 = WorkflowTimeTracker(db_path=db_path)
            start_time = tracker1.start_workflow("persistent_test")
            time.sleep(0.1)
            tracker1.end_workflow("persistent_test", start_time)
            
            # Second instance should load the data
            tracker2 = WorkflowTimeTracker(db_path=db_path)
            
            assert "persistent_test" in tracker2.data["workflows"]
            stats = tracker2.get_workflow_stats("persistent_test")
            assert stats["count"] == 1