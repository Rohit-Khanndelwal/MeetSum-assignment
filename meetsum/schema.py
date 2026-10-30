"""Structured output (E4): the meeting-summary schema, JSON extraction and validation."""
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, ValidationError, field_validator

from .data import speakers


class ActionItem(BaseModel):
    owner: str
    task: str
    due: Optional[str] = None      # as stated in the conversation ("tomorrow", "Friday"), or null


class MeetingSummary(BaseModel):
    summary: str
    participants: List[str]
    key_points: List[str]
    decisions: List[str]
    action_items: List[ActionItem]

    @field_validator("summary")
    @classmethod
    def _not_empty(cls, v):
        if not v.strip():
            raise ValueError("summary is empty")
        return v.strip()


FALLBACK_TEXT = "Summary unavailable: the model output could not be validated."


def fallback(dialogue):
    """Safe output when validation fails twice: no invented content, only the known speakers."""
    return MeetingSummary(summary=FALLBACK_TEXT, participants=speakers(dialogue),
                          key_points=[], decisions=[], action_items=[])


JSON_SCHEMA = MeetingSummary.model_json_schema()

SCHEMA_INSTRUCTION = (
    "Return ONLY a JSON object with exactly these keys:\n"
    '  "summary": the summary text,\n'
    '  "participants": the speaker names exactly as written in the conversation,\n'
    '  "key_points": a list of short strings, one per topic discussed,\n'
    '  "decisions": a list of things the speakers agreed or decided (empty list if none),\n'
    '  "action_items": a list of {"owner": name, "task": what they will do, "due": when as stated, or null} '
    "(empty list if none).\n"
    "Only include facts stated in the conversation. No markdown, no text before or after the JSON."
)

_GROUP_OWNERS = {"all", "everyone", "both", "team", "group", "they", "we"}


@dataclass
class ValidationResult:
    ok: bool
    response: Optional[MeetingSummary] = None
    stage: Optional[str] = None            # where it failed: "json", "schema" or "grounding"
    errors: list = field(default_factory=list)


def extract_json(text):
    """Return the first balanced {...} block in text, ignoring braces inside strings."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def check_grounding(resp, dialogue):
    """Participants must be real speakers; action-item owners must be named in the conversation."""
    errors = []
    spk = {s.lower() for s in speakers(dialogue)}
    low = dialogue.lower()
    fake = [p for p in resp.participants if p.strip().lower() not in spk]
    if fake:
        errors.append(f"participants not in the conversation: {fake}")
    for item in resp.action_items:
        owner = item.owner.strip().lower()
        if owner and owner not in _GROUP_OWNERS and owner not in low:
            errors.append(f"action-item owner '{item.owner}' is not named in the conversation")
    return errors


def parse_and_validate(raw, dialogue=None, grounding=True):
    blob = extract_json(raw)
    if blob is None:
        return ValidationResult(False, stage="json", errors=["no JSON object found"])
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        return ValidationResult(False, stage="json", errors=[f"invalid JSON: {e.msg}"])
    try:
        resp = MeetingSummary.model_validate(data)
    except ValidationError as e:
        msgs = [f"{'.'.join(map(str, err['loc'])) or 'object'}: {err['msg']}" for err in e.errors()]
        return ValidationResult(False, stage="schema", errors=msgs)
    if grounding and dialogue is not None:
        errs = check_grounding(resp, dialogue)
        if errs:
            return ValidationResult(False, response=resp, stage="grounding", errors=errs)
    return ValidationResult(True, response=resp)


def repair_message(errors):
    return ("Your previous output was invalid because: " + "; ".join(errors)
            + ".\n" + SCHEMA_INSTRUCTION)
