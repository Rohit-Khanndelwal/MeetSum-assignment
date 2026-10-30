"""Responsible-use controls applied before text reaches the model.

Chat transcripts often contain phone numbers and emails (SAMSum included), so PII is redacted
from the dialogue. Injection phrases inside a transcript are flagged, not obeyed; injection in the
user's own request is blocked.
"""
import re

_PII = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){8,11}\d(?!\d)"),
}

_INJECTION = [
    "ignore previous instructions", "ignore all previous", "ignore the above",
    "reveal your system prompt", "you are now", "disregard your instructions",
]


def redact_pii(text):
    found = []
    for kind, pat in _PII.items():
        if pat.search(text):
            found.append(kind)
            text = pat.sub(f"[{kind.upper()}]", text)
    return text, found


def detect_injection(text):
    low = text.lower()
    return [p for p in _INJECTION if p in low]


def check_dialogue(dialogue):
    clean, pii = redact_pii(dialogue)
    return clean, {"pii_redacted": pii, "injection_in_transcript": detect_injection(clean)}


def check_request(request):
    injection = detect_injection(request)
    return {"injection_in_request": injection, "blocked": bool(injection)}
