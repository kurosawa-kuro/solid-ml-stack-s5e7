"""
Comprehensive test cases for utility modules to achieve 95% coverage

Integrated enhanced test cases for complete coverage.
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
    notify_complete, 
    notify_error, 
    notify_start, 
    notify_submission
)
from src.util.time_tracker import (
    WorkflowTimer, 
    WorkflowTimeTracker, 
    time_workflow
)


class TestWebhookNotifierFull:
    """Comprehensive tests for WebhookNotifier"""

    def test_init_with_no_webhook(self):
        """Test initialization without webhook URL"""
        with patch.dict(os.environ, {}, clear=True):
            notifier = WebhookNotifier()
            assert notifier.webhook_url is None

    def test_init_with_env_webhook(self):
        """Test initialization with environment variable"""
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https: //discord.com/webhook"}):
            notifier = WebhookNotifier()
            assert notifier.webhook_url == "https: //discord.com/webhook"

    def test_init_with_direct_webhook(self):
        """Test initialization with direct URL"""
        notifier = WebhookNotifier(webhook_url="https: //custom.webhook")
        assert notifier.webhook_url == "https: //custom.webhook"

    @patch("requests.post")
    def test_send_message_no_webhook(self, mock_post):
        """Test sending message without webhook configured"""
        notifier = WebhookNotifier()
        result = notifier.send_message("Test message")

        assert result is False
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_send_message_success(self, mock_post):
        """Test successful message sending"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier(webhook_url="https: //test.webhook")
        result = notifier.send_message("Test message", username="TestBot")

        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https: //test.webhook"
        assert call_args[1]["json"]["content"] == "Test message"
        assert call_args[1]["json"]["username"] == "TestBot"

    @patch("requests.post")
    def test_send_message_with_embeds(self, mock_post):
        """Test sending message with embeds"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier(webhook_url="https: //test.webhook")
        embed = {"title": "Test Embed", "color": 123456}
        result = notifier.send_message("", embeds=[embed])

        assert result is True
        call_args = mock_post.call_args
        assert call_args[1]["json"]["embeds"] == [embed]

    @patch("requests.post")
    def test_send_message_failure(self, mock_post):
        """Test message sending failure"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        notifier = WebhookNotifier(webhook_url="https: //test.webhook")
        result = notifier.send_message("Test message")

        assert result is False

    @patch("requests.post")
    def test_send_message_exception(self, mock_post):
        """Test message sending with exception"""
        mock_post.side_effect = requests.RequestException("Network error")

        notifier = WebhookNotifier(webhook_url="https: //test.webhook")
        result = notifier.send_message("Test message")

        assert result is False

    @patch("src.util.notifications.WebhookNotifier.send_message")
    def test_notify_training_start(self, mock_send):
        """Test training start notification"""
        mock_send.return_value = True

        notifier = WebhookNotifier(webhook_url="https: //test.webhook")
        config = {"model": "LightGBM", "params": {"lr": 0.1}}
        result = notifier.notify_training_start("TestModel", config)

        assert result is True
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        embeds = call_args[1]["embeds"]
        assert len(embeds) == 1
        assert "Training Started" in embeds[0]["title"]
        assert "TestModel" in embeds[0]["title"]

    @patch("src.util.notifications.WebhookNotifier.send_message")
    def test_notify_training_complete(self, mock_send):
        """Test training completion notification"""
        mock_send.return_value = True

        notifier = WebhookNotifier(webhook_url="https: //test.webhook")
        metrics = {"accuracy": 0.95, "auc": 0.98}
        result = notifier.notify_training_complete("TestModel", metrics, duration=120.5)

        assert result is True
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        embeds = call_args[1]["embeds"]
        assert "Training Complete" in embeds[0]["title"]
        assert "120.50s" in str(embeds[0]["fields"])

    @patch("src.util.notifications.WebhookNotifier.send_message")
    def test_notify_error(self, mock_send):
        """Test error notification"""
        mock_send.return_value = True

        notifier = WebhookNotifier(webhook_url="https: //test.webhook")
        result = notifier.notify_error("data_loading", "File not found")

        assert result is True
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        embeds = call_args[1]["embeds"]
        assert "Error" in embeds[0]["title"]
        assert "data_loading" in embeds[0]["title"]

    @patch("src.util.notifications.WebhookNotifier.send_message")
    def test_notify_submission(self, mock_send):
        """Test submission notification"""
        mock_send.return_value = True

        notifier = WebhookNotifier(webhook_url="https: //test.webhook")
        result = notifier.notify_submission(score=0.975, rank=100, improvement=0.002)

        assert result is True
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        embeds = call_args[1]["embeds"]
        assert "Submission Result" in embeds[0]["title"]
        assert any("0.975" in str(field) for field in embeds[0]["fields"])


class TestNotificationConvenienceFunctions:
    """Test convenience notification functions"""

    @patch("src.util.notifications.WebhookNotifier")
    def test_notify_start_convenience(self, mock_notifier_class):
        """Test notify_start convenience function"""
        mock_instance = Mock()
        mock_instance.notify_training_start.return_value = True
        mock_notifier_class.return_value = mock_instance

        result = notify_start("TestModel", {"param": "value"})

        assert result is True
        mock_instance.notify_training_start.assert_called_once_with("TestModel", {"param": "value"})

    @patch("src.util.notifications.WebhookNotifier")
    def test_notify_start_exception(self, mock_notifier_class):
        """Test notify_start with exception"""
        mock_notifier_class.side_effect = Exception("Init error")

        result = notify_start("TestModel")

        assert result is False

    @patch("src.util.notifications.WebhookNotifier")
    def test_notify_complete_convenience(self, mock_notifier_class):
        """Test notify_complete convenience function"""
        mock_instance = Mock()
        mock_instance.notify_training_complete.return_value = True
        mock_notifier_class.return_value = mock_instance

        metrics = {"accuracy": 0.95}
        result = notify_complete("TestModel", metrics, duration=100)

        assert result is True
        mock_instance.notify_training_complete.assert_called_once()

    @patch("src.util.notifications.WebhookNotifier")
    def test_notify_error_convenience(self, mock_notifier_class):
        """Test notify_error convenience function"""
        mock_instance = Mock()
        mock_instance.notify_error.return_value = True
        mock_notifier_class.return_value = mock_instance

        result = notify_error("test_stage", "Test error")

        assert result is True
        mock_instance.notify_error.assert_called_once_with("test_stage", "Test error")

    @patch("src.util.notifications.WebhookNotifier")
    def test_notify_submission_convenience(self, mock_notifier_class):
        """Test notify_submission convenience function"""
        mock_instance = Mock()
        mock_instance.notify_submission.return_value = True
        mock_notifier_class.return_value = mock_instance

        result = notify_submission(0.98, rank=50)

        assert result is True
        mock_instance.notify_submission.assert_called_once_with(0.98, rank=50, improvement=None)


class TestWorkflowTimeTrackerCore:
    """Core tests for WorkflowTimeTracker basic functionality"""

    def test_init_default(self):
        """Test default initialization"""
        tracker = WorkflowTimeTracker()
        assert str(tracker.db_path) == "data/workflow_times.json"
        assert isinstance(tracker.data, dict)
        assert "workflows" in tracker.data

    def test_init_custom_path(self):
        """Test initialization with custom path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "custom_times.json")
            tracker = WorkflowTimeTracker(db_path)
            assert tracker.db_path == Path(db_path)

    def test_load_existing_data(self):
        """Test loading existing data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing data file
            db_path = os.path.join(tmpdir, "times.json")
            existing_data = {
                "workflows": {
                    "test_workflow": {
                        "executions": [{"timestamp": "2024-01-01", "duration": 10.5}],
                        "statistics": {"average": 10.5},
                    }
                }
            }
            with open(db_path, "w") as f:
                json.dump(existing_data, f)

            # Load tracker
            tracker = WorkflowTimeTracker(db_path)
            assert "test_workflow" in tracker.data["workflows"]
            assert tracker.data["workflows"]["test_workflow"]["statistics"]["average"] == 10.5

    def test_load_corrupted_data(self):
        """Test loading corrupted data file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "corrupted.json")
            with open(db_path, "w") as f:
                f.write("not valid json{")

            tracker = WorkflowTimeTracker(db_path)
            assert tracker.data == {"workflows": {}}

    def test_start_and_end_workflow(self):
        """Test workflow timing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(os.path.join(tmpdir, "times.json"))

            # Start workflow
            start_time = tracker.start_workflow("test_task")
            time.sleep(0.1)  # Simulate work

            # End workflow
            tracker.end_workflow("test_task", start_time)

            # Check data
            assert "test_task" in tracker.data["workflows"]
            executions = tracker.data["workflows"]["test_task"]["executions"]
            assert len(executions) == 1
            assert executions[0]["duration"] > 0.1

    def test_update_statistics(self):
        """Test statistics update"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(os.path.join(tmpdir, "times.json"))

            # Add multiple executions
            for duration in [1.0, 2.0, 3.0]:
                tracker.start_workflow("stats_test")
                time.sleep(0.01)
                tracker.data["workflows"]["stats_test"]["executions"][-1]["duration"] = duration
                tracker.update_statistics("stats_test")

            stats = tracker.get_workflow_stats("stats_test")
            assert stats["average"] == 2.0
            assert stats["median"] == 2.0
            assert stats["min"] == 1.0
            assert stats["max"] == 3.0
            assert stats["count"] == 3
            assert "std_dev" in stats

    def test_get_workflow_stats_nonexistent(self):
        """Test getting stats for non-existent workflow"""
        tracker = WorkflowTimeTracker()
        stats = tracker.get_workflow_stats("nonexistent")
        assert stats is None

    def test_get_estimated_duration(self):
        """Test duration estimation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(os.path.join(tmpdir, "times.json"))

            # No data
            estimate = tracker.get_estimated_duration("new_workflow")
            assert estimate is None

            # Add data
            tracker.data["workflows"]["test"] = {"statistics": {"average": 5.0, "std_dev": 0.5}}

            estimate = tracker.get_estimated_duration("test")
            assert estimate == 6.0  # average + 2*std_dev

    def test_list_workflows(self):
        """Test listing workflows"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(os.path.join(tmpdir, "times.json"))

            # Add workflows
            tracker.start_workflow("workflow1")
            tracker.start_workflow("workflow2")

            workflows = tracker.list_workflows()
            assert "workflow1" in workflows
            assert "workflow2" in workflows

    def test_clear_workflow_data(self):
        """Test clearing workflow data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(os.path.join(tmpdir, "times.json"))

            # Add and clear
            tracker.start_workflow("to_clear")
            assert "to_clear" in tracker.data["workflows"]

            result = tracker.clear_workflow_data("to_clear")
            assert result is True
            assert "to_clear" not in tracker.data["workflows"]

            # Clear non-existent
            result = tracker.clear_workflow_data("nonexistent")
            assert result is False

    def test_print_workflow_stats(self):
        """Test printing workflow stats"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(os.path.join(tmpdir, "times.json"))

            # Add workflow with stats
            tracker.data["workflows"]["test"] = {
                "statistics": {
                    "average": 5.0,
                    "median": 4.5,
                    "min": 3.0,
                    "max": 7.0,
                    "count": 10,
                }
            }

            # Should not raise exception
            tracker.print_workflow_stats("test")
            tracker.print_workflow_stats("nonexistent")

    def test_print_all_stats(self):
        """Test printing all stats"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(os.path.join(tmpdir, "times.json"))

            # Add multiple workflows
            for i in range(3):
                tracker.data["workflows"][f"workflow{i}"] = {"statistics": {"average": float(i + 1)}}

            # Should not raise exception
            tracker.print_all_stats()

    def test_execution_limit(self):
        """Test execution history limit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(os.path.join(tmpdir, "times.json"))

            # Add more than 100 executions
            for i in range(110):
                start = tracker.start_workflow("limit_test")
                tracker.end_workflow("limit_test", start)

            # Should keep only last 100
            executions = tracker.data["workflows"]["limit_test"]["executions"]
            assert len(executions) == 100


class TestWorkflowTimeTrackerModules:
    """Additional tests from test_util_modules.py integration"""

    def test_init_custom_path(self):
        """Test custom path initialization"""
        custom_path = "/tmp/custom_times.json"
        tracker = WorkflowTimeTracker(custom_path)
        assert str(tracker.db_path) == custom_path

    def test_start_workflow(self):
        """Test starting a workflow"""
        tracker = WorkflowTimeTracker()
        start_time = tracker.start_workflow("test_workflow")
        assert isinstance(start_time, float)
        assert start_time > 0

    def test_end_workflow(self):
        """Test ending a workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path)

            start_time = tracker.start_workflow("test_workflow")
            time.sleep(0.1)  # Small delay
            tracker.end_workflow("test_workflow", start_time)

            # Check that data was saved
            assert os.path.exists(db_path)
            assert "test_workflow" in tracker.data["workflows"]

    def test_list_workflows(self):
        """Test listing workflows"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path)

            # Add some workflows
            start1 = tracker.start_workflow("workflow1")
            tracker.end_workflow("workflow1", start1)
            start2 = tracker.start_workflow("workflow2")
            tracker.end_workflow("workflow2", start2)

            workflows = tracker.list_workflows()
            assert "workflow1" in workflows
            assert "workflow2" in workflows

    def test_get_workflow_stats(self):
        """Test getting workflow statistics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path)

            # Add multiple runs of same workflow
            for _ in range(3):
                start = tracker.start_workflow("test_workflow")
                time.sleep(0.05)
                tracker.end_workflow("test_workflow", start)

            stats = tracker.get_workflow_stats("test_workflow")
            assert stats is not None
            assert isinstance(stats, dict)

    def test_get_workflow_stats_nonexistent(self):
        """Test getting stats for nonexistent workflow"""
        tracker = WorkflowTimeTracker()
        stats = tracker.get_workflow_stats("nonexistent")
        assert stats is None

    def test_load_data_creates_structure(self):
        """Test that _load_data creates proper structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "new_times.json")
            tracker = WorkflowTimeTracker(db_path)
            data = tracker._load_data()
            assert "workflows" in data

    def test_save_data(self):
        """Test saving database"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path)

            # Add some data
            tracker.data = {"workflows": {"test": {"executions": []}}}
            tracker._save_data()

            # Load and verify
            with open(db_path, "r") as f:
                saved_data = json.load(f)
            assert "workflows" in saved_data


class TestWorkflowTimerAdditional:
    """Additional tests for WorkflowTimer from test_util_modules.py"""

    def test_workflow_timer_context_manager(self):
        """Test WorkflowTimer as context manager"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path)

            with WorkflowTimer(tracker, "test_context"):
                time.sleep(0.1)

            stats = tracker.get_workflow_stats("test_context")
            assert stats["count"] == 1
            assert stats["average"] >= 0.1

    def test_workflow_timer_exception_handling(self):
        """Test WorkflowTimer handles exceptions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_times.json")
            tracker = WorkflowTimeTracker(db_path)

            try:
                with WorkflowTimer(tracker, "test_exception"):
                    time.sleep(0.05)
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Should still record the time
            stats = tracker.get_workflow_stats("test_exception")
            assert stats["count"] == 1


class TestWorkflowTimer:
    """Tests for WorkflowTimer context manager"""

    def test_workflow_timer_basic(self):
        """Test basic timer usage"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(os.path.join(tmpdir, "times.json"))

            with WorkflowTimer(tracker, "context_test"):
                time.sleep(0.1)

            stats = tracker.get_workflow_stats("context_test")
            assert stats["count"] == 1
            assert stats["average"] > 0.1

    def test_workflow_timer_exception(self):
        """Test timer with exception"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WorkflowTimeTracker(os.path.join(tmpdir, "times.json"))

            with pytest.raises(ValueError):
                with WorkflowTimer(tracker, "error_test"):
                    raise ValueError("Test error")

            # Should still record time
            stats = tracker.get_workflow_stats("error_test")
            assert stats["count"] == 1


class TestTimeWorkflowDecoratorAdditional:
    """Additional decorator tests from test_util_modules.py"""

    def test_time_workflow_decorator(self):
        """Test time_workflow decorator"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_times.json")

            @time_workflow("decorated_func", db_path)
            def test_function(x, y):
                time.sleep(0.05)
                return x + y

            result = test_function(2, 3)
            assert result == 5

            # Check timing was recorded
            tracker = WorkflowTimeTracker(db_path)
            stats = tracker.get_workflow_stats("decorated_func")
            assert stats["count"] == 1
            assert stats["average"] >= 0.05

    def test_time_workflow_decorator_with_exception(self):
        """Test time_workflow decorator handles exceptions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_times.json")

            @time_workflow("exception_func", db_path)
            def failing_function():
                time.sleep(0.05)
                raise RuntimeError("Test error")

            with pytest.raises(RuntimeError):
                failing_function()

            # Should still record the time
            tracker = WorkflowTimeTracker(db_path)
            stats = tracker.get_workflow_stats("exception_func")
            assert stats["count"] == 1


class TestTimeWorkflowDecorator:
    """Tests for time_workflow decorator"""

    def test_time_workflow_decorator(self):
        """Test workflow timing decorator"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "times.json")

            @time_workflow("decorated_func", db_path)
            def test_function(x, y):
                time.sleep(0.1)
                return x + y

            result = test_function(1, 2)
            assert result == 3

            # Check timing was recorded
            tracker = WorkflowTimeTracker(db_path)
            stats = tracker.get_workflow_stats("decorated_func")
            assert stats["count"] == 1
            assert stats["average"] > 0.1

    def test_time_workflow_decorator_exception(self):
        """Test decorator with exception"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "times.json")

            @time_workflow("error_func", db_path)
            def failing_function():
                raise RuntimeError("Test error")

            with pytest.raises(RuntimeError):
                failing_function()

            # Should still record time
            tracker = WorkflowTimeTracker(db_path)
            stats = tracker.get_workflow_stats("error_func")
            assert stats["count"] == 1


class TestWebhookNotifierModules:
    """Additional webhook notifier tests from test_util_modules.py"""

    def test_init_with_url(self):
        """Test initialization with webhook URL"""
        notifier = WebhookNotifier("https://test.webhook.url")
        assert notifier.webhook_url == "https://test.webhook.url"

    def test_init_without_url_fails(self):
        """Test initialization without webhook URL fails"""
        with pytest.raises(ValueError):
            WebhookNotifier()

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://env.webhook.url"})
    def test_init_from_env(self):
        """Test initialization from environment variable"""
        notifier = WebhookNotifier()
        assert notifier.webhook_url == "https://env.webhook.url"

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://test.url"})
    def test_send_message_no_url_in_constructor(self):
        """Test sending message with URL from env"""
        notifier = WebhookNotifier()
        # This should have a webhook URL from env
        assert notifier.webhook_url == "https://test.url"

    @patch("requests.post")
    def test_send_message_success(self, mock_post):
        """Test successful message sending"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://test.webhook.url")
        result = notifier.send_message("test message")

        assert result is True
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_send_message_failure(self, mock_post):
        """Test message sending failure"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://test.webhook.url")
        result = notifier.send_message("test message")

        assert result is False

    @patch("requests.post")
    def test_send_message_exception(self, mock_post):
        """Test message sending with exception"""
        mock_post.side_effect = Exception("Network error")

        notifier = WebhookNotifier("https://test.webhook.url")
        result = notifier.send_message("test message")

        assert result is False

    @patch("requests.post")
    def test_notify_training_start(self, mock_post):
        """Test training start notification"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://test.webhook.url")
        config = {"model": "LightGBM", "features": 10}
        result = notifier.notify_training_start("TestModel", config)

        assert result is True
        # Check that proper message was sent
        call_args = mock_post.call_args
        assert "TestModel" in str(call_args)

    @patch("requests.post")
    def test_notify_training_complete(self, mock_post):
        """Test training completion notification"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://test.webhook.url")
        metrics = {"accuracy": 0.95, "auc": 0.98}
        result = notifier.notify_training_complete("TestModel", metrics, 120.5)

        assert result is True
        call_args = mock_post.call_args
        assert "TestModel" in str(call_args)
        assert "0.95" in str(call_args)

    @patch("requests.post")
    def test_notify_error(self, mock_post):
        """Test error notification"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://test.webhook.url")
        result = notifier.notify_error("training", "Test error message")

        assert result is True
        call_args = mock_post.call_args
        assert "error" in str(call_args).lower()

    @patch("requests.post")
    def test_notify_submission(self, mock_post):
        """Test submission notification"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://test.webhook.url")
        result = notifier.notify_submission(0.975, rank=100, improvement=0.005)

        assert result is True
        call_args = mock_post.call_args
        assert "0.975" in str(call_args)


class TestNotificationFunctionsModules:
    """Test notification helper functions from test_util_modules.py"""

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://test.webhook.url"})
    @patch("requests.post")
    def test_notify_start(self, mock_post):
        """Test notify_start function"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        config = {"model": "LightGBM", "features": 15}
        result = notify_start("TestExperiment", config)

        assert result is True
        mock_post.assert_called_once()

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://test.webhook.url"})
    @patch("requests.post")
    def test_notify_complete(self, mock_post):
        """Test notify_complete function"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        metrics = {"accuracy": 0.96, "auc": 0.99}
        result = notify_complete("TestExperiment", metrics, duration=180.7)

        assert result is True
        mock_post.assert_called_once()

    def test_notify_start_no_webhook(self):
        """Test notify_start without webhook URL"""
        with patch.dict(os.environ, {}, clear=True):
            result = notify_start("TestExperiment", {})
            assert result is False

    def test_notify_complete_no_webhook(self):
        """Test notify_complete without webhook URL"""
        with patch.dict(os.environ, {}, clear=True):
            result = notify_complete("TestExperiment", {}, 100)
            assert result is False


class TestWebhookNotifierEnhanced:
    """Enhanced tests for WebhookNotifier functionality"""

    def test_init_with_url_enhanced(self):
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


class TestWorkflowTimeTrackerEnhanced:
    """Enhanced tests for WorkflowTimeTracker functionality"""

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


class TestUtilitiesIntegrationEnhanced:
    """Enhanced tests for integration scenarios between utility modules"""

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


class TestEdgeCasesAndErrorHandlingEnhanced:
    """Enhanced tests for edge cases and error handling scenarios"""

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
