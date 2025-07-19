from unittest.mock import Mock, patch

from src.util.notifications import WebhookNotifier, notify_complete, notify_error, notify_start, notify_submission


class TestWebhookNotifier:
    """Test WebhookNotifier normal cases only."""

    @patch.dict("os.environ", {"WEBHOOK_DISCORD": "https://discord.com/api/webhooks/test"})
    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        notifier = WebhookNotifier()
        assert notifier.webhook_url == "https://discord.com/api/webhooks/test"

    def test_init_with_url_parameter(self):
        """Test initialization with URL parameter."""
        url = "https://discord.com/api/webhooks/custom"
        notifier = WebhookNotifier(webhook_url=url)
        assert notifier.webhook_url == url

    @patch("requests.post")
    def test_send_message_success(self, mock_post):
        """Test successful message sending."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        result = notifier.send_message("Test message")

        assert result is True
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_send_message_with_embeds(self, mock_post):
        """Test message sending with embeds."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        embeds = [{"title": "Test", "color": 123}]
        result = notifier.send_message("Test", embeds=embeds)

        assert result is True
        args, kwargs = mock_post.call_args
        assert "embeds" in kwargs["json"]

    @patch("requests.post")
    def test_notify_training_start(self, mock_post):
        """Test training start notification."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        config = {"learning_rate": 0.1, "num_leaves": 31, "n_estimators": 100}
        result = notifier.notify_training_start("LightGBM", config)

        assert result is True

    @patch("requests.post")
    def test_notify_training_complete(self, mock_post):
        """Test training completion notification."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        metrics = {"accuracy": 0.95, "f1": 0.93}
        result = notifier.notify_training_complete("XGBoost", metrics, 120.5)

        assert result is True

    @patch("requests.post")
    def test_notify_error(self, mock_post):
        """Test error notification."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        result = notifier.notify_error("training", "Memory error occurred")

        assert result is True

    @patch("requests.post")
    def test_notify_submission(self, mock_post):
        """Test submission notification."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        notifier = WebhookNotifier("https://discord.com/api/webhooks/test")
        result = notifier.notify_submission(0.975, rank=100, improvement=0.002)

        assert result is True


class TestConvenienceFunctions:
    """Test convenience functions normal cases."""

    @patch("src.util.notifications.WebhookNotifier")
    def test_notify_start(self, mock_notifier_class):
        """Test notify_start convenience function."""
        mock_notifier = Mock()
        mock_notifier.notify_training_start.return_value = True
        mock_notifier_class.return_value = mock_notifier

        result = notify_start("TestModel", {"param": "value"})

        assert result is True
        mock_notifier.notify_training_start.assert_called_once_with("TestModel", {"param": "value"})

    @patch("src.util.notifications.WebhookNotifier")
    def test_notify_complete(self, mock_notifier_class):
        """Test notify_complete convenience function."""
        mock_notifier = Mock()
        mock_notifier.notify_training_complete.return_value = True
        mock_notifier_class.return_value = mock_notifier

        metrics = {"accuracy": 0.96}
        result = notify_complete("TestModel", metrics, 100.0)

        assert result is True
        mock_notifier.notify_training_complete.assert_called_once_with("TestModel", metrics, 100.0)

    @patch("src.util.notifications.WebhookNotifier")
    def test_notify_error_function(self, mock_notifier_class):
        """Test notify_error convenience function."""
        mock_notifier = Mock()
        mock_notifier.notify_error.return_value = True
        mock_notifier_class.return_value = mock_notifier

        result = notify_error("validation", "Test error")

        assert result is True
        mock_notifier.notify_error.assert_called_once_with("validation", "Test error")

    @patch("src.util.notifications.WebhookNotifier")
    def test_notify_submission_function(self, mock_notifier_class):
        """Test notify_submission convenience function."""
        mock_notifier = Mock()
        mock_notifier.notify_submission.return_value = True
        mock_notifier_class.return_value = mock_notifier

        result = notify_submission(0.978, rank=50)

        assert result is True
        mock_notifier.notify_submission.assert_called_once_with(0.978, rank=50)
