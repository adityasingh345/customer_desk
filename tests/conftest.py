import pytest
# conftest.py is a special pytest file that applies to every test this one removes your rela keys during tests. without it once your .env has a key 

@pytest.fixture(autouse=True)
def no_real_api_keys(monkeypatch):
    """Tests must never call the real Gemini or Telegram, even if .env has keys."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)