# it decides where to send the classified message 
# this is the python version of n8n's switch by category + IF (by urgency) 

from .models import RouteDecision, ProcessedMessage

REPLIES = {
    "sales": "Thanks for your interest! Our team will get back to you shortly with details.",
    "support": "Thanks for reaching out. We've received your question and will reply soon.",
    "billing": "We've received your billing query. Our accounts team will look into it and get back to you.",
}

def route(processed: ProcessedMessage) -> RouteDecision:
    # if ai failed 
    c  = processed.classification
    if processed.needs_review:
        return RouteDecision(
            queue="support",
            reply=REPLIES["support"],
            alert_owner=True,
            reason="AI classification failed; flagged for human review",
        )
        
    if c.category == "spam":
        return RouteDecision(queue="spam", reply=None, alert_owner=False, reason="classified as spam")
    
    # everything else 
    return RouteDecision(
        queue=c.category,
        reply=REPLIES[c.category],
        alert_owner=(c.urgency == "high"),
        reason=f"{c.category} / {c.urgency} urgency",
    )