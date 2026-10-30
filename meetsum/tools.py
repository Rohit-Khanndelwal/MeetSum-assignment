"""Tool / function calling (E5): fetch a meeting transcript, and log action items as tasks.

Both tools are simulations backed by in-memory stores; no real calendar or task system is touched.
"""
import json
import re
from itertools import count

from .data import speakers
from .schema import extract_json

TOOL_SPECS = [
    {
        "name": "fetch_transcript",
        "description": "Fetch the transcript of a recorded meeting or chat by its ID.",
        "parameters": {"type": "object",
                       "properties": {"meeting_id": {"type": "string"}},
                       "required": ["meeting_id"]},
        "returns": {"meeting_id": "str", "dialogue": "str", "speakers": ["str"]},
    },
    {
        "name": "create_task",
        "description": "Log one action item in the team task tracker.",
        "parameters": {"type": "object",
                       "properties": {"owner": {"type": "string"},
                                      "task": {"type": "string"},
                                      "due": {"type": ["string", "null"]},
                                      "meeting_id": {"type": ["string", "null"]}},
                       "required": ["owner", "task"]},
        "returns": {"task_id": "str", "status": "created"},
    },
]
SPEC_BY_NAME = {s["name"]: s for s in TOOL_SPECS}

FETCH_SYSTEM_PROMPT = (
    "You are MeetSum, an assistant that summarizes meetings and chats.\n"
    "The user refers to a meeting by ID, so you must fetch its transcript first.\n"
    "You have this tool:\n"
    f"{json.dumps(SPEC_BY_NAME['fetch_transcript'], indent=2)}\n"
    "Reply with ONLY this JSON and nothing else:\n"
    '{"tool": "fetch_transcript", "arguments": {"meeting_id": "<id>"}}'
)


class TranscriptStore:
    def __init__(self, records):
        self.by_id = {r["id"]: r["dialogue"] for r in records}

    def fetch(self, meeting_id):
        dialogue = self.by_id.get(str(meeting_id))
        if dialogue is None:
            return {"error": f"no meeting with id {meeting_id}"}
        return {"meeting_id": str(meeting_id), "dialogue": dialogue, "speakers": speakers(dialogue)}


class TaskTracker:
    def __init__(self):
        self.tasks = []
        self._ids = count(1)

    def create(self, owner, task, due=None, meeting_id=None):
        t = {"task_id": f"T{next(self._ids):04d}", "owner": owner, "task": task, "due": due,
             "meeting_id": meeting_id, "status": "created"}
        self.tasks.append(t)
        return {"task_id": t["task_id"], "status": "created"}


_store = None
tracker = TaskTracker()


def set_store(store):
    global _store
    _store = store


def fetch_transcript(meeting_id):
    if _store is None:
        raise RuntimeError("Call tools.set_store(TranscriptStore(records)) first.")
    return _store.fetch(meeting_id)


def create_task(owner, task, due=None, meeting_id=None):
    return tracker.create(owner, task, due, meeting_id)


TOOLS = {"fetch_transcript": fetch_transcript, "create_task": create_task}


def parse_tool_call(text):
    """Return {"tool": name, "arguments": {...}} if text holds a valid call, else None."""
    blob = extract_json(text)
    if blob is None:
        return None
    try:
        call = json.loads(blob)
    except json.JSONDecodeError:
        return None
    name, args = call.get("tool"), call.get("arguments")
    if name not in TOOLS or not isinstance(args, dict):
        return None
    if any(not str(args.get(k, "")).strip() for k in SPEC_BY_NAME[name]["parameters"]["required"]):
        return None
    return {"tool": name, "arguments": args}


def execute_tool_call(call):
    return TOOLS[call["tool"]](**call["arguments"])


def find_meeting_id(text):
    m = re.search(r"\b\d{6,}(?:-\d+)?\b", text)
    return m.group(0) if m else None
