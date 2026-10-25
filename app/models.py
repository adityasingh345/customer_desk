# the one shape every customer message takes, no matter where it comes from 

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

#incoming messages is pydantic model every message must have 4 fields 
# literal means channel can only be one of those two values

class IncomingMessage(BaseModel):
    name: str
    message: str
    channel: Literal["telegram", "web_form"]
    chat_id: str = ""
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator("name")
    @classmethod
    def clean_name(cls, v:str) -> str:
        v = v.strip()
        return v or "unknown"
    
    @field_validator("message")
    @classmethod
    def clean_message(cls, v: str) -> str:
        v = v.strip()
        if not v :
            raise ValueError("mess should not be empty")
        if len(v) > 2000: 
            raise ValueError("message too long ")
        return v 
    
#----------------------------------phase2----
Category = Literal["sales", "support", "billing", "spam"]
Urgency = Literal["low", "medium", "high"]

class Classification(BaseModel):
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
    def trim_summary(cls, v: str)-> str:
        return v.strip()[:200]
    

class ProcessedMessage(BaseModel):
    message: IncomingMessage
    classification: Classification
    needs_review: bool = False