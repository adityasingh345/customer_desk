import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app import classifier
from app.classifier import ClassificationFailed, classify
from app.intake import handle_message, normalize_form
from app.models import Classification


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def no_sleep(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr(classifier.asyncio, "sleep", instant)


def msg(text="Can I get a quote?"):
    return normalize_form({"name": "Asha", "message": text})


def fake_gemini(monkeypatch, responses):
    """Replace the Gemini client. `responses` is a list of text-or-Exception, one per call."""
    calls = {"n": 0}

    async def generate_content(**kwargs):
        item = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)

    fake = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    monkeypatch.setattr(classifier, "_get_client", lambda: fake)
    return calls


GOOD = json.dumps({"category": "sales", "urgency": "low", "summary": "Wants a quote."})


# --- the data shape ---------------------------------------------------------

def test_classification_rejects_unknown_category():
    with pytest.raises(ValidationError):
        Classification(category="complaint", urgency="low", summary="x")


def test_classification_trims_long_summary():
    c = Classification(category="support", urgency="low", summary="x" * 500)
    assert len(c.summary) == 200


# --- classify(): parsing, retry, failure ------------------------------------

async def test_classify_parses_valid_json(monkeypatch):
    fake_gemini(monkeypatch, [GOOD])
    c = await classify(msg())
    assert (c.category, c.urgency) == ("sales", "low")


async def test_classify_retries_then_succeeds(monkeypatch, no_sleep):
    calls = fake_gemini(monkeypatch, [RuntimeError("429 rate limit"), "not json", GOOD])
    c = await classify(msg())
    assert c.category == "sales" and calls["n"] == 3


async def test_classify_gives_up_after_three_attempts(monkeypatch, no_sleep):
    calls = fake_gemini(monkeypatch, [RuntimeError("boom")])
    with pytest.raises(ClassificationFailed):
        await classify(msg())
    assert calls["n"] == 3


async def test_classify_rejects_out_of_schema_answer(monkeypatch, no_sleep):
    bad = json.dumps({"category": "angry", "urgency": "low", "summary": "x"})
    fake_gemini(monkeypatch, [bad])
    with pytest.raises(ClassificationFailed):
        await classify(msg())


# --- handle_message(): the pipeline ----------------------------------------

async def test_pipeline_saves_raw_and_classified(tmp_data):
    async def fake_classifier(m):
        return Classification(category="billing", urgency="high", summary="Double charge.")

    result = await handle_message(msg("I was charged twice"), classifier=fake_classifier)
    assert result.classification.category == "billing" and not result.needs_review
    assert (tmp_data / "messages.jsonl").exists()
    saved = json.loads((tmp_data / "classified.jsonl").read_text().strip())
    assert saved["classification"]["urgency"] == "high"


async def test_pipeline_survives_ai_failure(tmp_data):
    async def broken(m):
        raise RuntimeError("AI is down")

    result = await handle_message(msg(), classifier=broken)
    assert result.needs_review is True
    assert result.classification.category == "support"
    # the raw message was saved BEFORE the AI was called
    assert (tmp_data / "messages.jsonl").read_text().strip()


async def test_pipeline_without_api_key_skips_classification(tmp_data):
    result = await handle_message(msg())  # conftest removed GEMINI_API_KEY
    assert result is None
    assert (tmp_data / "messages.jsonl").exists()
    assert not (tmp_data / "classified.jsonl").exists()