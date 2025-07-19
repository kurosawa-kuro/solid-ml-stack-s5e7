import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


class WebhookNotifier:
    """Discord webhook notification utility for ML pipeline events."""

    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize webhook notifier.

        Args:
            webhook_url: Discord webhook URL. If None, reads from env var.
        """
        self.webhook_url = webhook_url or os.getenv("WEBHOOK_DISCORD")
        if not self.webhook_url:
            raise ValueError("Webhook URL not provided and " "WEBHOOK_DISCORD env var not set")

    def send_message(
        self,
        content: str,
        username: str = "ML Pipeline",
        embeds: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Send a basic message to Discord webhook.

        Args:
            content: Message content
            username: Bot username to display
            embeds: Optional Discord embeds for rich formatting

        Returns:
            True if successful, False otherwise
        """
        payload: Dict[str, Any] = {"content": content, "username": username}

        if embeds:
            payload["embeds"] = embeds

        try:
            if self.webhook_url:
                response = requests.post(self.webhook_url, json=payload)
                return response.status_code == 204
            return False
        except Exception as e:
            print(f"Webhook notification failed: {e}")
            return False

    def notify_training_start(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Notify training start with model configuration."""
        embed = {
            "title": f"🚀 Training Started: {model_name}",
            "color": 3447003,  # Blue
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Model", "value": model_name, "inline": True},
                {
                    "name": "Config",
                    "value": (f"```json\n" f"{json.dumps(config, indent=2)[: 1000]}```"),
                    "inline": False,
                },
            ],
        }
        return self.send_message("", embeds=[embed])

    def notify_training_complete(
        self,
        model_name: str,
        metrics: Dict[str, float],
        duration: float,
    ) -> bool:
        """Notify training completion with metrics."""
        metrics_text = "\n".join([f"{k}: {v: .6f}" for k, v in metrics.items()])

        embed = {
            "title": f"✅ Training Complete: {model_name}",
            "color": 65280,  # Green
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Model", "value": model_name, "inline": True},
                {
                    "name": "Duration",
                    "value": f"{duration: .2f}s",
                    "inline": True,
                },
                {
                    "name": "Metrics",
                    "value": f"```\n{metrics_text}```",
                    "inline": False,
                },
            ],
        }
        return self.send_message("", embeds=[embed])

    def notify_error(self, stage: str, error: str) -> bool:
        """Notify pipeline error."""
        embed = {
            "title": f"❌ Error in {stage}",
            "color": 16711680,  # Red
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {"name": "Stage", "value": stage, "inline": True},
                {
                    "name": "Error",
                    "value": f"```\n{error[: 1000]}```",
                    "inline": False,
                },
            ],
        }
        return self.send_message("", embeds=[embed])

    def notify_submission(
        self,
        score: float,
        rank: Optional[int] = None,
        improvement: Optional[float] = None,
    ) -> bool:
        """Notify submission results."""
        fields = [{"name": "Score", "value": f"{score: .6f}", "inline": True}]

        if rank:
            fields.append({"name": "Rank", "value": str(rank), "inline": True})

        if improvement:
            improvement_text = f"+{improvement: .6f}" if improvement > 0 else f"{improvement: .6f}"
            fields.append(
                {
                    "name": "Improvement",
                    "value": improvement_text,
                    "inline": True,
                }
            )

        color = 16776960  # Yellow for submission
        if improvement and improvement > 0:
            color = 65280  # Green for improvement

        embed = {
            "title": "🎯 Submission Result",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": fields,
        }
        return self.send_message("", embeds=[embed])


# Convenience functions for quick notifications
def notify_start(model_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """Quick notification for training start."""
    try:
        notifier = WebhookNotifier()
        return notifier.notify_training_start(model_name, config or {})
    except Exception:
        return False


def notify_complete(model_name: str, metrics: Dict[str, float], duration: float = 0) -> bool:
    """Quick notification for training completion."""
    try:
        notifier = WebhookNotifier()
        return notifier.notify_training_complete(model_name, metrics, duration)
    except Exception:
        return False


def notify_error(stage: str, error: str) -> bool:
    """Quick notification for errors."""
    try:
        notifier = WebhookNotifier()
        return notifier.notify_error(stage, str(error))
    except Exception:
        return False


def notify_submission(score: float, **kwargs: Any) -> bool:
    """Quick notification for submissions."""
    try:
        notifier = WebhookNotifier()
        return notifier.notify_submission(score, **kwargs)
    except Exception:
        return False
