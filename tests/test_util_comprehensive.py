"""
Comprehensive test cases for utility modules to achieve 95% coverage
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from src.util.notifications import WebhookNotifier, notify_complete, notify_error, notify_start, notify_submission
from src.util.time_tracker import WorkflowTimer, WorkflowTimeTracker, time_workflow


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


class TestWorkflowTimeTracker:
    """Comprehensive tests for WorkflowTimeTracker"""

    def test_init_default(self):
        """Test default initialization"""
        tracker = WorkflowTimeTracker()
        assert tracker.db_path == Path("data/workflow_times.json")
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
