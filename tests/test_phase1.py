import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.intake import normalize_form, normalize_telegram
from app.main import app


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    return tmp_path


def test_telegram_and_form_have_same_shape():
    t = normalize_telegram("Asha", "What are your timings?", 12345)
    f = normalize_form({"name": "Ravi", "message": "Do you take walk-ins?"})
    assert t.model_dump().keys() == f.model_dump().keys()
    assert (t.channel, t.chat_id) == ("telegram", "12345")
    assert (f.channel, f.chat_id) == ("web_form", "")


def test_missing_first_name_becomes_unknown():
    assert normalize_telegram(None, "hi", 1).name == "Unknown"


def test_empty_message_rejected():
    with pytest.raises(ValidationError):
        normalize_form({"name": "A", "message": "   "})


def test_too_long_message_rejected():
    with pytest.raises(ValidationError):
        normalize_form({"name": "A", "message": "x" * 2001})


def test_contact_endpoint_saves_message(tmp_data):
    with TestClient(app) as client:
        r = client.post("/contact", json={"name": "Asha", "message": "Hello"})
    assert r.status_code == 200 and r.json() == {"status": "received"}
    line = (tmp_data / "messages.jsonl").read_text().strip()
    saved = json.loads(line)
    assert saved["name"] == "Asha" and saved["channel"] == "web_form"


def test_contact_endpoint_rejects_blank(tmp_data):
    with TestClient(app) as client:
        r = client.post("/contact", json={"name": "Asha", "message": "  "})
    assert r.status_code == 422
    assert not (tmp_data / "messages.jsonl").exists()


def test_form_page_served():
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200 and "<form" in r.text