"""
Test cases for util modules to improve coverage
"""

import json
import os
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from src.util.notifications import WebhookNotifier, notify_complete, notify_start
from src.util.time_tracker import WorkflowTimer, WorkflowTimeTracker, time_workflow


class TestWorkflowTimeTracker:
    """Test WorkflowTimeTracker class"""

    def test_init_default(self):
        """Test default initialization"""
        tracker = WorkflowTimeTracker()
        assert str(tracker.db_path) == "data/workflow_times.json"

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


class TestWorkflowTimer:
    """Test WorkflowTimer context manager"""

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


class TestTimeWorkflowDecorator:
    """Test time_workflow decorator"""

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


class TestWebhookNotifier:
    """Test WebhookNotifier class"""

    def test_init_with_url(self):
        """Test initialization with webhook URL"""
        notifier = WebhookNotifier("https: //test.webhook.url")
        assert notifier.webhook_url == "https: //test.webhook.url"

    def test_init_without_url_fails(self):
        """Test initialization without webhook URL fails"""
        with pytest.raises(ValueError):
            WebhookNotifier()

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https: //env.webhook.url"})
    def test_init_from_env(self):
        """Test initialization from environment variable"""
        notifier = WebhookNotifier()
        assert notifier.webhook_url == "https: //env.webhook.url"

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https: //test.url"})
    def test_send_message_no_url_in_constructor(self):
        """Test sending message with URL from env"""
        notifier = WebhookNotifier()
        # This should have a webhook URL from env
        assert notifier.webhook_url == "https: //test.url"

    @patch("requests.post")
    def test_send_message_success(self, mock_post):
        """Test successful message sending"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https: //test.webhook.url")
        result = notifier.send_message("test message")

        assert result is True
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_send_message_failure(self, mock_post):
        """Test message sending failure"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https: //test.webhook.url")
        result = notifier.send_message("test message")

        assert result is False

    @patch("requests.post")
    def test_send_message_exception(self, mock_post):
        """Test message sending with exception"""
        mock_post.side_effect = Exception("Network error")

        notifier = WebhookNotifier("https: //test.webhook.url")
        result = notifier.send_message("test message")

        assert result is False

    @patch("requests.post")
    def test_notify_training_start(self, mock_post):
        """Test training start notification"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https: //test.webhook.url")
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

        notifier = WebhookNotifier("https: //test.webhook.url")
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

        notifier = WebhookNotifier("https: //test.webhook.url")
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

        notifier = WebhookNotifier("https: //test.webhook.url")
        result = notifier.notify_submission(0.975, rank=100, improvement=0.005)

        assert result is True
        call_args = mock_post.call_args
        assert "0.975" in str(call_args)


class TestNotificationFunctions:
    """Test notification helper functions"""

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https: //test.webhook.url"})
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

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https: //test.webhook.url"})
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
