"""
Notification System Integration Tests
Tests for webhook notifications with proper environment variable handling and mocking
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import requests

from src.util.notifications import (
    WebhookNotifier,
    notify_complete,
    notify_error,
    notify_start,
    notify_submission,
)


class TestWebhookEnvironmentHandling:
    """Test environment variable handling for webhook notifications"""

    def test_webhook_url_from_environment(self):
        """Test webhook URL loading from environment variable"""
        test_url = "https://discord.com/api/webhooks/test"
        
        with patch.dict(os.environ, {"WEBHOOK_DISCORD": test_url}):
            notifier = WebhookNotifier()
            assert notifier.webhook_url == test_url

    def test_webhook_url_from_parameter(self):
        """Test webhook URL loading from parameter overrides environment"""
        test_url = "https://discord.com/api/webhooks/test"
        env_url = "https://discord.com/api/webhooks/env"
        
        with patch.dict(os.environ, {"WEBHOOK_DISCORD": env_url}):
            notifier = WebhookNotifier(webhook_url=test_url)
            assert notifier.webhook_url == test_url

    def test_missing_webhook_url_raises_error(self):
        """Test that missing webhook URL raises appropriate error"""
        # Clear environment variable if it exists
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Webhook URL not provided"):
                WebhookNotifier()

    def test_empty_webhook_url_raises_error(self):
        """Test that empty webhook URL raises appropriate error"""
        with patch.dict(os.environ, {"WEBHOOK_DISCORD": ""}):
            with pytest.raises(ValueError, match="Webhook URL not provided"):
                WebhookNotifier()

    def test_none_webhook_url_with_env_works(self):
        """Test that None webhook_url falls back to environment"""
        test_url = "https://discord.com/api/webhooks/test"
        
        with patch.dict(os.environ, {"WEBHOOK_DISCORD": test_url}):
            notifier = WebhookNotifier(webhook_url=None)
            assert notifier.webhook_url == test_url


class TestWebhookNotificationMocking:
    """Test webhook notifications with proper mocking"""

    def setup_method(self):
        """Setup for each test method"""
        self.test_url = "https://discord.com/api/webhooks/test"
        self.notifier = WebhookNotifier(webhook_url=self.test_url)

    @patch("requests.post")
    def test_successful_message_sending(self, mock_post):
        """Test successful message sending with proper response"""
        # Mock successful Discord response
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        result = self.notifier.send_message("Test message")
        
        assert result is True
        mock_post.assert_called_once()
        
        # Verify request parameters
        args, kwargs = mock_post.call_args
        assert args[0] == self.test_url
        assert "json" in kwargs
        assert kwargs["json"]["content"] == "Test message"
        assert kwargs["json"]["username"] == "ML Pipeline"

    @patch("requests.post")
    def test_failed_message_sending(self, mock_post):
        """Test failed message sending with error response"""
        # Mock failed Discord response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        result = self.notifier.send_message("Test message")
        
        assert result is False
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_network_error_handling(self, mock_post):
        """Test network error handling"""
        # Mock network error
        mock_post.side_effect = requests.RequestException("Network error")

        result = self.notifier.send_message("Test message")
        
        assert result is False
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_training_start_notification(self, mock_post):
        """Test training start notification format"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        config = {"learning_rate": 0.1, "n_estimators": 100}
        result = self.notifier.notify_training_start("LightGBM", config)
        
        assert result is True
        
        # Verify embed structure
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        
        embed = payload["embeds"][0]
        assert "🚀 Training Started: LightGBM" in embed["title"]
        assert embed["color"] == 3447003  # Blue
        assert "fields" in embed
        
        # Check fields
        fields = embed["fields"]
        model_field = next(f for f in fields if f["name"] == "Model")
        config_field = next(f for f in fields if f["name"] == "Config")
        
        assert model_field["value"] == "LightGBM"
        assert "learning_rate" in config_field["value"]

    @patch("requests.post")
    def test_training_complete_notification(self, mock_post):
        """Test training complete notification format"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        metrics = {"accuracy": 0.95, "auc": 0.98}
        result = self.notifier.notify_training_complete("LightGBM", metrics, 120.5)
        
        assert result is True
        
        # Verify embed structure
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        embed = payload["embeds"][0]
        
        assert "✅ Training Complete: LightGBM" in embed["title"]
        assert embed["color"] == 65280  # Green
        
        # Check fields
        fields = embed["fields"]
        duration_field = next(f for f in fields if f["name"] == "Duration")
        metrics_field = next(f for f in fields if f["name"] == "Metrics")
        
        assert "120.50s" in duration_field["value"]
        assert "accuracy" in metrics_field["value"]
        assert "0.950000" in metrics_field["value"]

    @patch("requests.post")
    def test_error_notification(self, mock_post):
        """Test error notification format"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        result = self.notifier.notify_error("training", "Out of memory error")
        
        assert result is True
        
        # Verify embed structure
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        embed = payload["embeds"][0]
        
        assert "❌ Error in training" in embed["title"]
        assert embed["color"] == 16711680  # Red
        
        # Check fields
        fields = embed["fields"]
        error_field = next(f for f in fields if f["name"] == "Error")
        
        assert "Out of memory error" in error_field["value"]

    @patch("requests.post")
    def test_submission_notification_with_improvement(self, mock_post):
        """Test submission notification with improvement"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        result = self.notifier.notify_submission(0.976, rank=50, improvement=0.002)
        
        assert result is True
        
        # Verify embed structure
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        embed = payload["embeds"][0]
        
        assert "🎯 Submission Result" in embed["title"]
        assert embed["color"] == 65280  # Green for improvement
        
        # Check fields
        fields = embed["fields"]
        score_field = next(f for f in fields if f["name"] == "Score")
        rank_field = next(f for f in fields if f["name"] == "Rank")
        improvement_field = next(f for f in fields if f["name"] == "Improvement")
        
        assert "0.976000" in score_field["value"]
        assert "50" in rank_field["value"]
        assert "0.002000" in improvement_field["value"]  # May have space formatting

    @patch("requests.post")
    def test_submission_notification_no_improvement(self, mock_post):
        """Test submission notification without improvement"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        result = self.notifier.notify_submission(0.974, improvement=-0.001)
        
        assert result is True
        
        # Verify embed structure
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        embed = payload["embeds"][0]
        
        assert embed["color"] == 16776960  # Yellow for no improvement


class TestConvenienceFunctions:
    """Test convenience notification functions"""

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_notify_start_convenience(self, mock_post):
        """Test notify_start convenience function"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        config = {"model": "test"}
        result = notify_start("TestModel", config)
        
        assert result is True
        mock_post.assert_called_once()

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_notify_complete_convenience(self, mock_post):
        """Test notify_complete convenience function"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        metrics = {"accuracy": 0.95}
        result = notify_complete("TestModel", metrics, 100.0)
        
        assert result is True
        mock_post.assert_called_once()

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_notify_error_convenience(self, mock_post):
        """Test notify_error convenience function"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        result = notify_error("testing", "Test error")
        
        assert result is True
        mock_post.assert_called_once()

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_notify_submission_convenience(self, mock_post):
        """Test notify_submission convenience function"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        result = notify_submission(0.95, rank=10, improvement=0.01)
        
        assert result is True
        mock_post.assert_called_once()

    def test_convenience_functions_without_env_var(self):
        """Test convenience functions fail gracefully without environment variable"""
        with patch.dict(os.environ, {}, clear=True):
            # All should return False when webhook URL is not available
            assert notify_start("TestModel") is False
            assert notify_complete("TestModel", {"acc": 0.9}) is False
            assert notify_error("test", "error") is False
            assert notify_submission(0.95) is False


class TestNotificationEdgeCases:
    """Test notification system edge cases"""

    def test_empty_webhook_url_send_message(self):
        """Test sending message with empty webhook URL"""
        notifier = WebhookNotifier(webhook_url="https://test.webhook")
        notifier.webhook_url = None  # Simulate empty URL
        
        result = notifier.send_message("Test")
        assert result is False

    @patch("requests.post")
    def test_long_config_truncation(self, mock_post):
        """Test that long configs are truncated properly"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier(webhook_url="https://test.webhook")
        
        # Create a very long config
        long_config = {f"param_{i}": f"value_{i}" * 100 for i in range(50)}
        
        result = notifier.notify_training_start("TestModel", long_config)
        assert result is True
        
        # Verify it was called and didn't fail
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_long_error_message_truncation(self, mock_post):
        """Test that long error messages are truncated"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier(webhook_url="https://test.webhook")
        
        # Create a very long error message
        long_error = "Error: " + "x" * 2000
        
        result = notifier.notify_error("testing", long_error)
        assert result is True
        
        # Verify the message was truncated
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        embed = payload["embeds"][0]
        error_field = next(f for f in embed["fields"] if f["name"] == "Error")
        
        assert len(error_field["value"]) <= 1010  # 1000 chars + markdown formatting

    @patch("requests.post")
    def test_unicode_message_handling(self, mock_post):
        """Test handling of unicode characters in messages"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier(webhook_url="https://test.webhook")
        
        unicode_message = "Model训练完成 🎉 Accuracy: 95%"
        result = notifier.send_message(unicode_message)
        
        assert result is True
        
        # Verify unicode was preserved
        args, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["content"] == unicode_message


class TestNotificationIntegrationScenarios:
    """Test notification integration scenarios"""

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_full_training_notification_workflow(self, mock_post):
        """Test complete training notification workflow"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        # Simulate full training workflow notifications
        config = {"model": "LightGBM", "cv_folds": 5}
        
        # 1. Start notification
        start_result = notify_start("LightGBM_CV", config)
        assert start_result is True
        
        # 2. Complete notification
        metrics = {"cv_score": 0.956, "cv_std": 0.012, "train_time": 245.6}
        complete_result = notify_complete("LightGBM_CV", metrics, 245.6)
        assert complete_result is True
        
        # 3. Submission notification
        submission_result = notify_submission(0.958, rank=45, improvement=0.003)
        assert submission_result is True
        
        # Verify all notifications were sent
        assert mock_post.call_count == 3

    @patch.dict(os.environ, {"WEBHOOK_DISCORD": "https://test.webhook"})
    @patch("requests.post")
    def test_error_recovery_notification_workflow(self, mock_post):
        """Test error and recovery notification workflow"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        # 1. Start training
        start_result = notify_start("TestModel", {"test": True})
        assert start_result is True
        
        # 2. Error occurs
        error_result = notify_error("training", "CUDA out of memory")
        assert error_result is True
        
        # 3. Recovery and completion
        recovery_result = notify_complete("TestModel_CPU", {"accuracy": 0.94}, 300.0)
        assert recovery_result is True
        
        assert mock_post.call_count == 3

    def test_notification_system_robustness(self):
        """Test notification system handles various failure modes gracefully"""
        # Test with various edge cases that should not crash
        notifier = WebhookNotifier(webhook_url="https://httpbin.org/status/404")
        
        # These should all return False but not crash
        assert notifier.send_message("") is False  # Empty message
        assert notifier.notify_training_start("", {}) is False  # Empty model name
        assert notifier.notify_training_complete("Model", {}, -1) is False  # Negative duration
        assert notifier.notify_error("", "") is False  # Empty error
        assert notifier.notify_submission(-1) is False  # Invalid score