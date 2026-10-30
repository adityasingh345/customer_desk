"""The data shapes used across the whole project.

Phase 1: IncomingMessage  - the ONE shape every customer message takes.
Phase 2: Classification   - what the AI tells us about a message.
         ProcessedMessage - a message plus its classification, ready to route.
Phase 3: RouteDecision    - where the message goes and what we do about it.
Phase 4: SupportAnswer    - the AI's answer to a support question, from the FAQ (RAG).
"""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class IncomingMessage(BaseModel):
    name: str
    message: str
    channel: Literal["telegram", "web_form"]
    chat_id: str = ""  # Telegram chat id; empty for the web form
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        v = v.strip()
        return v or "Unknown"

    @field_validator("message")
    @classmethod
    def clean_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > 2000:
            raise ValueError("message too long (max 2000 characters)")
        return v


# ---------------------------------------------------------------- Phase 2 ----

Category = Literal["sales", "support", "billing", "spam"]
Urgency = Literal["low", "medium", "high"]


class Classification(BaseModel):
    """The JSON shape we force the AI to answer in.

    The Field descriptions are sent to Gemini as part of the schema,
    so they act as extra instructions for each field.
    """

    category: Category = Field(
        description="sales = wants to buy/book/get a quote; support = questions or problems; "
        "billing = payments, invoices, refunds, charges; spam = ads, scams, gibberish."
    )
    urgency: Urgency = Field(
        description="high = angry, time-critical or safety-related; medium = normal; low = casual."
    )
    summary: str = Field(description="One short sentence (max 20 words) saying what the customer wants.")

    @field_validator("summary")
    @classmethod
    def trim_summary(cls, v: str) -> str:
        return v.strip()[:200]


# ---------------------------------------------------------------- Phase 3 ----


class RouteDecision(BaseModel):
    """What we decided to do with a classified message."""

    queue: Category  # which team's queue it goes to
    reply: str | None = None  # text to send back to the customer (None = say nothing)
    alert_owner: bool = False  # ping the owner on Telegram?
    reason: str  # why we decided this (great for debugging)
    sources: list[str] = Field(default_factory=list)  # Phase 4: FAQ chunk ids behind a RAG answer


class ProcessedMessage(BaseModel):
    message: IncomingMessage
    classification: Classification
    needs_review: bool = False  # True when the AI failed and a human should look
    decision: RouteDecision | None = None  # filled in by the router (Phase 3)


# ---------------------------------------------------------------- Phase 4 ----


class SupportAnswer(BaseModel):
    """The JSON shape we force the AI to answer support questions in (RAG)."""

    answerable: bool = Field(
        description="True ONLY if the context passages contain what is needed to answer the question."
    )
    answer: str = Field(
        description="Short, friendly answer (max 3 sentences) using ONLY the context. Empty string if not answerable."
    )
    sources: list[str] = Field(
        description="Ids of the context passages used, exactly as written in square brackets. Empty if not answerable."
    )