"""FastAPI service: triage inbound customer emails, store them as tickets, alert the team.

Run:
    uvicorn app.main:app --reload
Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.classifier import Classifier, build_classifier
from app.config import Settings, get_settings
from app.db import Ticket, make_session_factory, utcnow
from app.notifier import Notifier, format_alert
from app.schemas import (
    SLA_HOURS, Category, EmailIn, Priority, Stats, Status, StatusUpdate, TicketOut, Triage,
)

STATIC = Path(__file__).parent / "static"
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def create_app(
    settings: Settings | None = None,
    classifier: Classifier | None = None,
    notifier: Notifier | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    classifier = classifier or build_classifier(settings)
    notifier = notifier or Notifier(settings.chat_webhook_url)
    SessionLocal = make_session_factory(settings.database_url)

    app = FastAPI(
        title="AI Customer Email Triage API",
        description="Classifies customer emails by priority, category, sentiment and churn risk, "
        "tracks SLAs, and sends instant alerts to Google Chat / Slack.",
        version="1.0.0",
    )

    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")

    def _create_ticket(email: EmailIn, db: Session) -> Ticket:
        triage: Triage = classifier.classify(email)
        now = utcnow()
        sla = SLA_HOURS[triage.priority]
        ticket = Ticket(
            sender=str(email.sender),
            subject=email.subject,
            body=email.body,
            account_name=email.account_name,
            priority=triage.priority.value,
            category=triage.category.value,
            sentiment=triage.sentiment.value,
            churn_risk=triage.churn_risk,
            summary=triage.summary,
            suggested_reply=triage.suggested_reply,
            confidence=triage.confidence,
            engine=triage.engine,
            status=Status.open.value,
            sla_hours=sla,
            created_at=now,
            due_at=now + timedelta(hours=sla),
        )
        db.add(ticket)
        db.commit()
        if notifier.should_alert(triage):
            text = format_alert(ticket.id, email.subject, str(email.sender), email.account_name, triage)
            ticket.alert_sent = notifier.send(text)
            db.commit()
        return ticket

    @app.get("/", include_in_schema=False)
    def inbox_page():
        return FileResponse(STATIC / "index.html")

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "engine": "llm" if settings.llm_enabled else "rules",
                "alerts": bool(settings.chat_webhook_url)}

    @app.post("/triage/preview", response_model=Triage, tags=["triage"], dependencies=[Depends(require_key)])
    def preview(email: EmailIn):
        """Classify an email without saving it."""
        return classifier.classify(email)

    @app.post("/tickets", response_model=TicketOut, status_code=201, tags=["tickets"],
              dependencies=[Depends(require_key)])
    def create_ticket(email: EmailIn, db: Session = Depends(get_db)):
        """Triage an email, store it as a ticket and alert the team if it's high/medium priority."""
        return _create_ticket(email, db)

    @app.post("/tickets/batch", response_model=list[TicketOut], status_code=201, tags=["tickets"],
              dependencies=[Depends(require_key)])
    def create_batch(emails: list[EmailIn], db: Session = Depends(get_db)):
        if len(emails) > 100:
            raise HTTPException(status_code=413, detail="Max 100 emails per batch")
        return [_create_ticket(e, db) for e in emails]

    @app.get("/tickets", response_model=list[TicketOut], tags=["tickets"], dependencies=[Depends(require_key)])
    def list_tickets(
        priority: Priority | None = None,
        category: Category | None = None,
        status: Status | None = None,
        churn_risk: bool | None = None,
        limit: int = Query(50, ge=1, le=500),
        db: Session = Depends(get_db),
    ):
        """Open work queue, sorted by priority then oldest first."""
        stmt = select(Ticket)
        if priority:
            stmt = stmt.where(Ticket.priority == priority.value)
        if category:
            stmt = stmt.where(Ticket.category == category.value)
        if status:
            stmt = stmt.where(Ticket.status == status.value)
        if churn_risk is not None:
            stmt = stmt.where(Ticket.churn_risk == churn_risk)
        tickets = db.scalars(stmt).all()
        tickets = sorted(tickets, key=lambda t: (PRIORITY_ORDER[t.priority], t.created_at))
        return tickets[:limit]

    @app.get("/tickets/{ticket_id}", response_model=TicketOut, tags=["tickets"], dependencies=[Depends(require_key)])
    def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
        ticket = db.get(Ticket, ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return ticket

    @app.patch("/tickets/{ticket_id}", response_model=TicketOut, tags=["tickets"],
               dependencies=[Depends(require_key)])
    def update_status(ticket_id: int, update: StatusUpdate, db: Session = Depends(get_db)):
        ticket = db.get(Ticket, ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        ticket.status = update.status.value
        ticket.resolved_at = utcnow() if update.status == Status.resolved else None
        db.commit()
        return ticket

    @app.get("/stats", response_model=Stats, tags=["reporting"], dependencies=[Depends(require_key)])
    def stats(db: Session = Depends(get_db)):
        tickets = db.scalars(select(Ticket)).all()
        open_t = [t for t in tickets if t.status != Status.resolved.value]
        resolved = [t for t in tickets if t.status == Status.resolved.value]
        return Stats(
            total=len(tickets),
            open=len(open_t),
            by_priority=dict(Counter(t.priority for t in tickets)),
            by_category=dict(Counter(t.category for t in tickets)),
            by_sentiment=dict(Counter(t.sentiment for t in tickets)),
            churn_risk_open=sum(t.churn_risk for t in open_t),
            sla_breached_open=sum(t.sla_breached for t in open_t),
            sla_met_pct=round(100 * sum(not t.sla_breached for t in resolved) / len(resolved), 1)
            if resolved else None,
        )

    return app


app = create_app()
