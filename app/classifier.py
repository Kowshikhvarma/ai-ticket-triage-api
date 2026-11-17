"""Email triage engines.

* ``RuleBasedClassifier``: deterministic keyword + sentiment rules. Free, instant,
  works offline, and is the safety net when the LLM is unavailable.
* ``LLMClassifier``: asks an OpenAI-compatible chat model for structured JSON,
  validates it, and falls back to the rules on any error.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.schemas import Category, EmailIn, Priority, Sentiment, Triage

log = logging.getLogger(__name__)


class Classifier(Protocol):
    def classify(self, email: EmailIn) -> Triage: ...


# --------------------------------------------------------------------------- rules

CATEGORY_KEYWORDS: dict[Category, list[str]] = {
    Category.cancellation_risk: [
        "cancel", "terminate", "not renew", "won't renew", "switch to", "competitor",
        "alternative", "refund", "end our contract", "moving away", "churn",
    ],
    Category.bug: [
        "error", "bug", "broken", "crash", "not working", "doesn't work", "down", "outage",
        "500", "failed", "failing", "failure", "issue with", "can't load", "cannot load",
    ],
    Category.account_access: [
        "log in", "login", "password", "locked out", "access", "sso", "2fa", "reset link",
    ],
    Category.billing: ["invoice", "billing", "charged", "payment", "credit card", "gst", "receipt", "pricing"],
    Category.onboarding: [
        "onboarding", "set up", "setup", "getting started", "training", "kickoff", "import our data",
        "how do i", "walkthrough",
    ],
    Category.feature_request: ["feature request", "would be great if", "can you add", "wish", "suggestion", "roadmap"],
}

URGENT_WORDS = [
    "urgent", "asap", "immediately", "critical", "production", "blocking", "blocked",
    "down", "outage", "escalate", "deadline today", "right now", "all users", "nobody can",
]
NEGATIVE_WORDS = [
    "frustrated", "disappointed", "unacceptable", "angry", "terrible", "worst", "again",
    "still not", "third time", "no response", "poor", "unhappy", "annoyed", "waste",
]
POSITIVE_WORDS = ["thank", "great", "love", "awesome", "appreciate", "happy", "excellent", "helpful"]

REPLY_TEMPLATES: dict[Category, str] = {
    Category.cancellation_risk: (
        "Thank you for being candid with us. I'd like to understand what hasn't worked and what it would take "
        "to make this right. Could we get 20 minutes on a call in the next day or two?"
    ),
    Category.bug: (
        "Thanks for flagging this, and sorry for the disruption. I've raised it with our engineering team as a "
        "priority and will update you within {sla} hours. Could you share a screenshot and the time it happened?"
    ),
    Category.account_access: (
        "Sorry you're having trouble getting in. I'm looking into your account now and will update you within "
        "{sla} hours. In the meantime, please try the password reset link from the login page."
    ),
    Category.billing: (
        "Thanks for reaching out about billing. I've passed this to our finance team and will confirm the "
        "details within {sla} hours."
    ),
    Category.onboarding: (
        "Happy to help you get set up! I'll send over the relevant guide and can book a short walkthrough "
        "session. What times work for you this week?"
    ),
    Category.feature_request: (
        "Thank you for the suggestion, this is really useful. I've shared it with our product team and "
        "will keep you posted on where it lands on the roadmap."
    ),
    Category.general: "Thanks for your message. I'm looking into it and will get back to you within {sla} hours.",
}


def _hits(text: str, words: list[str]) -> list[str]:
    return [w for w in words if re.search(rf"\b{re.escape(w)}", text)]


class RuleBasedClassifier:
    """Transparent keyword scoring. Every decision lists the words that triggered it."""

    def classify(self, email: EmailIn) -> Triage:
        text = f"{email.subject}\n{email.body}".lower()
        reasons: list[str] = []

        cat_scores = {cat: _hits(text, kws) for cat, kws in CATEGORY_KEYWORDS.items()}
        best_cat, best_hits = max(cat_scores.items(), key=lambda kv: len(kv[1]))
        category = best_cat if best_hits else Category.general
        if best_hits:
            reasons.append(f"category '{category.value}' matched: {', '.join(best_hits[:4])}")

        neg, pos = _hits(text, NEGATIVE_WORDS), _hits(text, POSITIVE_WORDS)
        if len(neg) > len(pos):
            sentiment = Sentiment.negative
        elif pos and not neg:
            sentiment = Sentiment.positive
        else:
            sentiment = Sentiment.neutral
        if neg:
            reasons.append(f"negative tone: {', '.join(neg[:4])}")

        urgent = _hits(text, URGENT_WORDS)
        churn_risk = category == Category.cancellation_risk or (
            sentiment == Sentiment.negative and len(neg) >= 2
        )

        if urgent or category == Category.cancellation_risk:
            priority = Priority.high
        elif category in (Category.bug, Category.account_access, Category.billing) or sentiment == Sentiment.negative:
            priority = Priority.medium
        else:
            priority = Priority.low
        if urgent:
            reasons.append(f"urgency words: {', '.join(urgent[:4])}")
        if churn_risk:
            reasons.append("flagged as churn risk")

        signal = len(best_hits) + len(urgent) + len(neg)
        confidence = round(min(0.95, 0.45 + 0.1 * signal), 2)

        first_sentence = re.split(r"(?<=[.!?])\s+", email.body.strip())[0]
        summary = f"{email.subject.strip()}: {first_sentence[:160]}"
        sla = {Priority.high: 2, Priority.medium: 8, Priority.low: 24}[priority]
        greeting = f"Hi {email.account_name} team," if email.account_name else "Hi,"
        reply = f"{greeting}\n\n{REPLY_TEMPLATES[category].format(sla=sla)}\n\nBest regards,\nCustomer Success Team"

        return Triage(
            priority=priority,
            category=category,
            sentiment=sentiment,
            churn_risk=churn_risk,
            summary=summary,
            suggested_reply=reply,
            confidence=confidence,
            reasons=reasons,
            engine="rules",
        )


# --------------------------------------------------------------------------- LLM

SYSTEM_PROMPT = """You are a senior Customer Success Manager triaging inbound customer emails for a B2B SaaS company.
Return ONLY a JSON object with these keys:
  "priority":        "high" | "medium" | "low"
  "category":        "cancellation_risk" | "bug" | "billing" | "onboarding" | "feature_request" | "account_access" | "general"
  "sentiment":       "positive" | "neutral" | "negative"
  "churn_risk":      true | false
  "summary":         one sentence, max 30 words, what the customer needs
  "suggested_reply": a short, warm, professional first reply (max 90 words) signed "Customer Success Team"
  "confidence":      number 0-1
  "reasons":         list of 1-3 short strings explaining the priority

Priority rules:
- high: production outage, many users blocked, hard deadline today, cancellation/renewal threat, legal/security concern, or an angry repeat complaint.
- medium: a bug or access/billing issue affecting some users, or a clearly unhappy customer.
- low: how-to questions, feature requests, thank-you notes, general updates.
Never promise refunds, discounts or fix dates in the reply."""


class LLMClassifier:
    def __init__(self, settings: Settings, client: httpx.Client | None = None, fallback: Classifier | None = None):
        self.settings = settings
        self.client = client or httpx.Client(timeout=30)
        self.fallback = fallback or RuleBasedClassifier()

    def _call(self, email: EmailIn) -> dict:
        user_msg = (
            f"Account: {email.account_name or 'unknown'}\nFrom: {email.sender}\n"
            f"Subject: {email.subject}\n\n{email.body}"
        )
        resp = self.client.post(
            f"{self.settings.openai_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={
                "model": self.settings.openai_model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            },
        )
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])

    def classify(self, email: EmailIn) -> Triage:
        try:
            data = self._call(email)
            data["engine"] = "llm"
            return Triage.model_validate(data)
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, ValidationError) as exc:
            log.warning("LLM triage failed (%s); falling back to rules", exc.__class__.__name__)
            result = self.fallback.classify(email)
            result.reasons.append(f"LLM unavailable ({exc.__class__.__name__}), used rules")
            return result


def build_classifier(settings: Settings) -> Classifier:
    return LLMClassifier(settings) if settings.llm_enabled else RuleBasedClassifier()
