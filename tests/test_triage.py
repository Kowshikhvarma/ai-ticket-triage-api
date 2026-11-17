import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.classifier import LLMClassifier, RuleBasedClassifier
from app.config import Settings
from app.main import create_app
from app.notifier import Notifier, format_alert
from app.schemas import Category, EmailIn, Priority, Sentiment

OUTAGE = {
    "sender": "priya@acmehealth.com",
    "account_name": "Acme Health",
    "subject": "Production is down - nobody can log in",
    "body": "Since 9am none of our 40 users can log in. This is blocking our payroll run today.",
}
HOWTO = {
    "sender": "neha@cedar.edu",
    "subject": "How do I import our data?",
    "body": "We're getting started and want a walkthrough of the import.",
}


class RecordingNotifier(Notifier):
    def __init__(self):
        super().__init__(webhook_url="https://chat.example/webhook")
        self.sent: list[str] = []

    def send(self, text: str) -> bool:
        self.sent.append(text)
        return True


@pytest.fixture
def notifier():
    return RecordingNotifier()


@pytest.fixture
def client(tmp_path, notifier):
    settings = Settings(openai_api_key=None, database_url=f"sqlite:///{tmp_path}/t.db", api_key=None)
    app = create_app(settings, classifier=RuleBasedClassifier(), notifier=notifier)
    return TestClient(app)


# ---------------------------------------------------------------- rules engine

def test_rules_outage_is_high_priority():
    t = RuleBasedClassifier().classify(EmailIn(**OUTAGE))
    assert t.priority == Priority.high
    assert t.sla_hours == 2
    assert t.engine == "rules"
    assert any("urgency" in r for r in t.reasons)


def test_rules_cancellation_flags_churn_risk():
    t = RuleBasedClassifier().classify(EmailIn(
        sender="cfo@x.com", subject="Not renewing", body="We will cancel and move to a competitor."))
    assert t.category == Category.cancellation_risk
    assert t.churn_risk and t.priority == Priority.high


def test_rules_positive_note_is_low():
    t = RuleBasedClassifier().classify(EmailIn(
        sender="a@b.com", subject="Thanks!", body="Thank you, the team really appreciate it."))
    assert t.priority == Priority.low and t.sentiment == Sentiment.positive


def test_sample_set_accuracy():
    from scripts.evaluate import evaluate
    report = evaluate(RuleBasedClassifier())
    assert report["priority_accuracy"] >= 0.9
    assert report["category_accuracy"] >= 0.75


# ---------------------------------------------------------------- LLM engine

def _llm(handler) -> LLMClassifier:
    settings = Settings(openai_api_key="test-key")
    return LLMClassifier(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_llm_parses_structured_json():
    payload = {
        "priority": "high", "category": "bug", "sentiment": "negative", "churn_risk": False,
        "summary": "All users locked out, payroll blocked.", "suggested_reply": "Hi, on it.",
        "confidence": 0.9, "reasons": ["outage"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})

    t = _llm(handler).classify(EmailIn(**OUTAGE))
    assert t.engine == "llm" and t.priority == Priority.high


def test_llm_falls_back_to_rules_on_error():
    t = _llm(lambda r: httpx.Response(500)).classify(EmailIn(**OUTAGE))
    assert t.engine == "rules"
    assert any("LLM unavailable" in r for r in t.reasons)


def test_llm_falls_back_on_invalid_json_values():
    bad = {"priority": "super-urgent"}
    t = _llm(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(bad)}}]})
             ).classify(EmailIn(**OUTAGE))
    assert t.engine == "rules"


# ---------------------------------------------------------------- notifier

def test_alert_text_contains_key_facts():
    t = RuleBasedClassifier().classify(EmailIn(**OUTAGE))
    text = format_alert(7, OUTAGE["subject"], OUTAGE["sender"], "Acme Health", t)
    assert "HIGH priority" in text and "#7" in text and "Acme Health" in text


def test_notifier_without_webhook_does_not_send():
    assert Notifier(None).send("hello") is False


def test_notifier_posts_json_text():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200)

    n = Notifier("https://chat.example/hook", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert n.send("hi") is True and seen["body"] == {"text": "hi"}


# ---------------------------------------------------------------- API

def test_create_ticket_and_alert(client, notifier):
    r = client.post("/tickets", json=OUTAGE)
    assert r.status_code == 201
    body = r.json()
    assert body["priority"] == "high" and body["alert_sent"] is True and body["status"] == "open"
    assert len(notifier.sent) == 1


def test_low_priority_does_not_alert(client, notifier):
    assert client.post("/tickets", json=HOWTO).json()["priority"] == "low"
    assert notifier.sent == []


def test_queue_sorted_by_priority_and_filters(client):
    client.post("/tickets/batch", json=[HOWTO, OUTAGE])
    queue = client.get("/tickets").json()
    assert [t["priority"] for t in queue] == ["high", "low"]
    assert len(client.get("/tickets", params={"priority": "low"}).json()) == 1


def test_resolve_and_stats(client):
    tid = client.post("/tickets", json=OUTAGE).json()["id"]
    client.post("/tickets", json=HOWTO)
    r = client.patch(f"/tickets/{tid}", json={"status": "resolved"})
    assert r.json()["resolved_at"] is not None
    stats = client.get("/stats").json()
    assert stats["total"] == 2 and stats["open"] == 1 and stats["sla_met_pct"] == 100.0


def test_validation_and_404(client):
    assert client.post("/tickets", json={"sender": "not-an-email", "subject": "x", "body": "y"}).status_code == 422
    assert client.get("/tickets/999").status_code == 404


def test_api_key_required_when_configured(tmp_path):
    settings = Settings(openai_api_key=None, database_url=f"sqlite:///{tmp_path}/k.db", api_key="secret")
    c = TestClient(create_app(settings, classifier=RuleBasedClassifier(), notifier=Notifier(None)))
    assert c.post("/tickets", json=OUTAGE).status_code == 401
    assert c.post("/tickets", json=OUTAGE, headers={"X-API-Key": "secret"}).status_code == 201


def test_inbox_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Triage Inbox" in r.text
