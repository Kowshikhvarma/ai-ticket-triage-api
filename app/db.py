"""Ticket storage with SQLAlchemy (SQLite by default, PostgreSQL via DATABASE_URL)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sender: Mapped[str] = mapped_column(String(320))
    subject: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)
    account_name: Mapped[str | None] = mapped_column(String(200), nullable=True)

    priority: Mapped[str] = mapped_column(String(10), index=True)
    category: Mapped[str] = mapped_column(String(30), index=True)
    sentiment: Mapped[str] = mapped_column(String(10))
    churn_risk: Mapped[bool] = mapped_column(Boolean, default=False)
    summary: Mapped[str] = mapped_column(Text)
    suggested_reply: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    engine: Mapped[str] = mapped_column(String(10))

    status: Mapped[str] = mapped_column(String(15), default="open", index=True)
    sla_hours: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    due_at: Mapped[datetime] = mapped_column(DateTime)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    alert_sent: Mapped[bool] = mapped_column(Boolean, default=False)

    @property
    def sla_breached(self) -> bool:
        reference = self.resolved_at or utcnow()
        return reference > self.due_at


def make_session_factory(database_url: str) -> sessionmaker:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
