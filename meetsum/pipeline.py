"""The integrated MeetSum system (E7).

request ("Summarize meeting 13818513 and log its action items") or a pasted transcript
  -> request safety check
  -> no transcript? LLM emits a fetch_transcript tool call -> transcript store
  -> transcript safety check (PII redaction, injection flagged)
  -> prompt template -> LLM
  -> schema + grounding validation -> one repair retry -> safe fallback
  -> each validated action item -> create_task tool call
  -> final JSON
"""
from . import tools
from .prompts import PromptTemplate
from .schema import fallback, parse_and_validate, repair_message

FINAL_TEMPLATE = PromptTemplate("final", role="secretary", n_shots=0, detail="concise",
                                faithful=True, output_format="json")


def default_llm(messages, **kw):
    from .llm import generate
    return generate(messages, **kw)


def fetch_via_tool(request, llm):
    """Let the model emit the fetch call; fall back to an ID found in the request if it does not."""
    raw = llm([{"role": "system", "content": tools.FETCH_SYSTEM_PROMPT},
               {"role": "user", "content": request}], max_new_tokens=60)
    call = tools.parse_tool_call(raw)
    step = {"step": "tool_decision", "model_output": raw, "valid_call": bool(call and call["tool"] == "fetch_transcript")}
    if not step["valid_call"]:
        call = {"tool": "fetch_transcript", "arguments": {"meeting_id": tools.find_meeting_id(request)}}
    result = tools.execute_tool_call(call)
    step.update(call=call, result={k: v for k, v in result.items() if k != "dialogue"})
    return result, step


def summarize_meeting(request=None, dialogue=None, llm=None, template=FINAL_TEMPLATE, fewshot=(),
                      max_retries=1, decoding=None, safety=True, log_tasks=True):
    """Return {"response", "used_fetch_tool", "tasks_created", "valid_first_try", "retries", "fallback", "trace"}."""
    llm = llm or default_llm
    decoding = decoding or {}
    trace, meeting_id = [], None

    if safety and request:
        from .safety import check_request
        report = check_request(request)
        trace.append({"step": "request_check", **report})
        if report["blocked"]:
            return _result(fallback(dialogue or ""), trace, False, [], False, 0, True, dialogue)

    used_tool = dialogue is None
    if used_tool:
        result, step = fetch_via_tool(request or "", llm)
        trace.append(step)
        if "error" in result:
            return _result(fallback(""), trace, True, [], False, 0, True, None)
        dialogue, meeting_id = result["dialogue"], result["meeting_id"]

    if safety:
        from .safety import check_dialogue
        dialogue, report = check_dialogue(dialogue)
        trace.append({"step": "transcript_check", **report})

    messages = template.render(dialogue, fewshot)
    raw = llm(messages, **decoding)
    check = parse_and_validate(raw, dialogue)
    trace.append({"step": "generate", "attempt": 0, "model_output": raw, "valid": check.ok,
                  "stage": check.stage, "errors": check.errors})
    first_ok, retries = check.ok, 0

    while not check.ok and retries < max_retries:
        retries += 1
        messages = messages + [{"role": "assistant", "content": raw},
                               {"role": "user", "content": repair_message(check.errors)}]
        raw = llm(messages, **decoding)
        check = parse_and_validate(raw, dialogue)
        trace.append({"step": "repair", "attempt": retries, "model_output": raw, "valid": check.ok,
                      "stage": check.stage, "errors": check.errors})

    response = check.response if check.ok else fallback(dialogue)
    tasks = []
    if log_tasks and check.ok:
        for item in response.action_items:
            call = {"tool": "create_task", "arguments": {**item.model_dump(), "meeting_id": meeting_id}}
            tasks.append({"call": call, "result": tools.execute_tool_call(call)})
        trace.append({"step": "create_tasks", "calls": tasks})
    return _result(response, trace, used_tool, tasks, first_ok, retries, not check.ok, dialogue)


def _result(response, trace, used_tool, tasks, first, retries, is_fallback, dialogue):
    return {"response": response.model_dump(), "used_fetch_tool": used_tool, "tasks_created": tasks,
            "valid_first_try": first, "retries": retries, "fallback": is_fallback,
            "dialogue": dialogue, "trace": trace}
