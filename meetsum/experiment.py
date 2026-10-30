"""Experiment runners (E1-E7) and the shared experiment log."""
import csv
import uuid
from datetime import date
from itertools import combinations

from .config import LOG_PATH
from .metrics import rouge, score
from .prompts import PROMPTS, text_to_summary
from .schema import parse_and_validate

DECODING = {
    "D1": {"do_sample": False},
    "D2": {"do_sample": True, "temperature": 0.3},
    "D3": {"do_sample": True, "temperature": 0.7},
    "D4": {"do_sample": True, "temperature": 1.2},
    "D5": {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
    "D6": {"do_sample": True, "temperature": 0.7, "top_k": 50},
}

LOG_FIELDS = ["run_id", "date", "owner", "experiment", "prompt_id", "decoding_id", "fewshot_mode", "seed",
              "example_id", "output_raw", "prediction", "json_valid", "grounded", "rouge1", "rouge2",
              "rougeL", "words", "unsupported_names", "notes"]


def _llm(llm):
    if llm is None:
        from .llm import generate
        return generate
    return llm


def to_prediction(raw, template, dialogue):
    """Return (summary_text, json_valid, grounded, structured_or_None) for one raw reply."""
    if template.output_format != "json":
        return text_to_summary(raw), None, None, None
    check = parse_and_validate(raw, dialogue)
    if check.response is None:
        return "", False, None, None
    return check.response.summary, True, check.ok, check.response.model_dump()


def run_prompt(examples, prompt="P0", decoding="D1", seed=42, fewshot=(), llm=None,
               fewshot_mode="fixed", retriever=None, max_new_tokens=256):
    """Run one prompt x decoding config over examples. Returns (rows, metrics).

    fewshot_mode: "fixed" (the frozen few-shot file) or "retrieved" (most similar train dialogues).
    """
    llm = _llm(llm)
    template = PROMPTS[prompt] if isinstance(prompt, str) else prompt
    dec = DECODING[decoding] if isinstance(decoding, str) else decoding
    rows = []
    for ex in examples:
        shots = retriever.top(ex["dialogue"], template.n_shots) if fewshot_mode == "retrieved" else fewshot
        raw = llm(template.render(ex["dialogue"], shots), max_new_tokens=max_new_tokens, seed=seed, **dec)
        pred, valid, grounded, structured = to_prediction(raw, template, ex["dialogue"])
        rows.append({"example_id": ex["id"], "dialogue": ex["dialogue"], "reference": ex["summary"],
                     "output_raw": raw, "prediction": pred, "json_valid": valid, "grounded": grounded,
                     "structured": structured, "prompt_id": getattr(template, "name", "custom"),
                     "decoding_id": decoding if isinstance(decoding, str) else "custom",
                     "fewshot_mode": fewshot_mode if template.n_shots else "none", "seed": seed})
    return rows, score(rows)


def self_rouge_l(runs):
    """Mean pairwise ROUGE-L between seeds for the same inputs (100 = identical outputs, lower = more variable)."""
    vals = []
    for preds in zip(*[[r["prediction"] for r in rows] for rows in runs]):
        for a, b in combinations(preds, 2):
            vals.append(rouge(a, b)["rougeL"])
    return round(100 * sum(vals) / len(vals), 1) if vals else 100.0


def log_rows(rows, experiment, owner, notes=""):
    """Append scored rows to logs/experiment_log.csv (score() must have run first)."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_PATH.exists()
    run_id = uuid.uuid4().hex[:8]
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({**r, "run_id": run_id, "date": date.today().isoformat(), "owner": owner,
                        "experiment": experiment, "notes": notes,
                        "unsupported_names": ";".join(r.get("unsupported_names", []))})
    return run_id
