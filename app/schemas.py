"""Request / response models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Priority(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Category(str, Enum):
    cancellation_risk = "cancellation_risk"
    bug = "bug"
    billing = "billing"
    onboarding = "onboarding"
    feature_request = "feature_request"
    account_access = "account_access"
    general = "general"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class Status(str, Enum):
    open = "open"
    in_progress = "in_progress"
    resolved = "resolved"


SLA_HOURS = {Priority.high: 2, Priority.medium: 8, Priority.low: 24}


class EmailIn(BaseModel):
    sender: EmailStr
    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=20_000)
    account_name: str | None = Field(default=None, description="Customer account, if known")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sender": "priya@acmehealth.com",
                "subject": "Production is down - nobody can log in",
                "body": "Since 9am none of our 40 users can log in. This is blocking our payroll run today. Please help urgently.",
                "account_name": "Acme Health",
            }
        }
    )


class Triage(BaseModel):
    priority: Priority
    category: Category
    sentiment: Sentiment
    churn_risk: bool
    summary: str
    suggested_reply: str
    confidence: float = Field(ge=0, le=1)
    reasons: list[str] = []
    engine: str = Field(description="'llm' or 'rules'")

    @property
    def sla_hours(self) -> int:
        return SLA_HOURS[self.priority]


class TicketOut(BaseModel):
    id: int
    sender: str
    subject: str
    account_name: str | None
    priority: Priority
    category: Category
    sentiment: Sentiment
    churn_risk: bool
    summary: str
    suggested_reply: str
    confidence: float
    engine: str
    status: Status
    sla_hours: int
    created_at: datetime
    due_at: datetime
    resolved_at: datetime | None
    alert_sent: bool
    sla_breached: bool

    model_config = ConfigDict(from_attributes=True)


class StatusUpdate(BaseModel):
    status: Status


class Stats(BaseModel):
    total: int
    open: int
    by_priority: dict[str, int]
    by_category: dict[str, int]
    by_sentiment: dict[str, int]
    churn_risk_open: int
    sla_breached_open: int
    sla_met_pct: float | None = Field(description="Share of resolved tickets closed within SLA")
