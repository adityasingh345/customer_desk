# a tiny html page
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

from .intake import handle_message, normalize_form

router = APIRouter()

class ContactForm(BaseModel):
    name: str
    message: str

FORM_HTML = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Contact us</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:480px;margin:3rem auto;padding:0 1rem}
  input,textarea,button{width:100%;padding:.6rem;margin:.4rem 0;font:inherit;box-sizing:border-box}
  button{cursor:pointer}
</style>
<h2>How can we help?</h2>
<form id="f">
  <input name="name" placeholder="Your name" required>
  <textarea name="message" rows="5" placeholder="Your message" required></textarea>
  <button>Send</button>
</form>
<p id="status"></p>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const s = document.getElementById('status');
  const r = await fetch('/contact', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.fromEntries(new FormData(e.target)))
  });
  s.textContent = r.ok ? 'Thanks! We got your message.' : 'Something went wrong. Please try again.';
  if (r.ok) e.target.reset();
};
</script>
"""

@router.get("/", response_class=HTMLResponse)
async def form_page() -> str:
    return FORM_HTML


@router.post("/contact")
async def contact(form: ContactForm) -> dict:
    try:
        msg = normalize_form(form.model_dump())
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors(include_url=False, include_context=False))
    await handle_message(msg)
    return {"status": "received"}