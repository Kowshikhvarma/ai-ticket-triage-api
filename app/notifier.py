"""Send instant alerts for high and medium priority tickets.

Works with Google Chat incoming webhooks and Slack incoming webhooks: both
accept a JSON body with a ``text`` field.
"""

from __future__ import annotations

import logging

import httpx

from app.schemas import Priority, Triage

log = logging.getLogger(__name__)

ICONS = {Priority.high: "🔴", Priority.medium: "🟠", Priority.low: "🟢"}
ALERT_PRIORITIES = {Priority.high, Priority.medium}


def format_alert(ticket_id: int, subject: str, sender: str, account: str | None, triage: Triage) -> str:
    lines = [
        f"{ICONS[triage.priority]} *{triage.priority.value.upper()} priority* ticket #{ticket_id}"
        f" · respond within {triage.sla_hours}h",
        f"*Account:* {account or 'unknown'}  |  *From:* {sender}",
        f"*Subject:* {subject}",
        f"*Category:* {triage.category.value}  |  *Sentiment:* {triage.sentiment.value}",
        f"*Summary:* {triage.summary}",
    ]
    if triage.churn_risk:
        lines.append("⚠️ *Churn risk*: loop in the account owner.")
    return "\n".join(lines)


class Notifier:
    def __init__(self, webhook_url: str | None, client: httpx.Client | None = None):
        self.webhook_url = webhook_url
        self.client = client or httpx.Client(timeout=10)

    def should_alert(self, triage: Triage) -> bool:
        return triage.priority in ALERT_PRIORITIES

    def send(self, text: str) -> bool:
        """Post to the webhook. Returns True if delivered. Never raises."""
        if not self.webhook_url:
            log.info("No CHAT_WEBHOOK_URL set; alert not sent:\n%s", text)
            return False
        try:
            self.client.post(self.webhook_url, json={"text": text}).raise_for_status()
            return True
        except httpx.HTTPError as exc:
            log.error("Alert delivery failed: %s", exc)
            return False
