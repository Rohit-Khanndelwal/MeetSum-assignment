"""Evaluation harness (requirement 12): ROUGE plus length, faithfulness and structure checks."""
import re

from .config import CONCISE_MAX_WORDS
from .data import speakers

_scorer = None

# Capitalised words that are not names, so they never count as hallucinated entities.
_NOT_NAMES = set("""The A An She He They It This That These Those There Their Them His Her Hers Its We Our You
Your I Both All Everyone Someone Nobody Then When While After Before Because But And Or So If As At In On For To
From With Without By Of Summary Conversation Meeting Today Tomorrow Yesterday Tonight Monday Tuesday Wednesday
Thursday Friday Saturday Sunday January February March April May June July August September October November
December Mr Mrs Ms Dr According However Since Later Afterwards Instead Also Finally Meanwhile Eventually
Unfortunately Apparently Otherwise Extra Todays Christmas Easter""".split())


def rouge(pred, ref):
    global _scorer
    if _scorer is None:
        from rouge_score import rouge_scorer
        _scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = _scorer.score(ref, pred)
    return {k: v.fmeasure for k, v in s.items()}


def unsupported_names(summary, dialogue):
    """Capitalised words in the summary that never appear in the dialogue: likely hallucinated entities."""
    low = dialogue.lower()
    words = set(re.findall(r"\b[A-Z][a-z]{2,}\b", summary))
    return sorted(w for w in words if w not in _NOT_NAMES and w.lower() not in low)


def speaker_coverage(summary, dialogue):
    spk = speakers(dialogue)
    if not spk:
        return None
    low = summary.lower()
    return sum(s.lower() in low for s in spk) / len(spk)


def _mean(xs, pct=True):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    return round(100 * m, 1) if pct else round(m, 1)


def score_row(r):
    pred = r["prediction"]
    r.update(rouge(pred, r["reference"]))
    r["words"] = len(pred.split())
    r["unsupported_names"] = unsupported_names(pred, r["dialogue"])
    r["speaker_coverage"] = speaker_coverage(pred, r["dialogue"])
    return r


def score(rows, max_words=CONCISE_MAX_WORDS):
    """rows: dicts with 'prediction', 'reference', 'dialogue', optionally 'json_valid', 'grounded'."""
    for r in rows:
        score_row(r)
    return {
        "n": len(rows),
        "ROUGE-1": _mean([r["rouge1"] for r in rows]),
        "ROUGE-2": _mean([r["rouge2"] for r in rows]),
        "ROUGE-L": _mean([r["rougeL"] for r in rows]),
        "avg_words": _mean([r["words"] for r in rows], pct=False),
        "within_length_pct": _mean([float(r["words"] <= max_words) for r in rows]),
        "hallucinated_name_pct": _mean([float(bool(r["unsupported_names"])) for r in rows]),
        "speaker_coverage": _mean([r["speaker_coverage"] for r in rows]),
        "json_validity": _mean([r.get("json_valid") for r in rows]),
        "grounding_rate": _mean([r.get("grounded") for r in rows]),
    }


def categorize_failure(r, max_words=CONCISE_MAX_WORDS):
    """Assign one failure category to a scored row, or None if it looks acceptable."""
    if r.get("json_valid") is False:
        return "invalid_json"
    if r.get("grounded") is False:
        return "wrong_attribution"
    if r["unsupported_names"]:
        return "hallucinated_entity"
    if r["rougeL"] < 0.2:
        return "missed_content"
    if r["words"] > 1.5 * max_words:
        return "too_long"
    return None
