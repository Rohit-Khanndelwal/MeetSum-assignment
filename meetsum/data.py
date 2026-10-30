"""SAMSum loading, the frozen evaluation set and the few-shot demonstrations."""
import json
import random
import re
from collections import Counter

from .config import DATASET_ID, EVAL_SET_PATH, FEWSHOT_PATH, N_EVAL, N_FEWSHOT, SEED

_SPEAKER = re.compile(r"^\s*([^:\n]{1,40}):", re.M)


def speakers(dialogue):
    """Distinct speaker names in order of first appearance."""
    seen = []
    for name in _SPEAKER.findall(dialogue):
        name = name.strip()
        if name not in seen:
            seen.append(name)
    return seen


def to_record(ex):
    dialogue = (ex["dialogue"] or "").replace("\r\n", "\n").strip()
    return {
        "id": str(ex["id"]),
        "dialogue": dialogue,
        "summary": (ex["summary"] or "").strip(),
        "speakers": speakers(dialogue),
        "n_turns": len([ln for ln in dialogue.split("\n") if ln.strip()]),
        "n_words": len(dialogue.split()),
    }

def show_dataset():
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, split=split)
    return ds

def load_samsum(split="test"):
    """Return one SAMSum split as a list of flat records, dropping empty dialogues."""
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, split=split)
    return [r for r in (to_record(ex) for ex in ds) if r["dialogue"] and r["summary"]]


def length_bucket(n_turns):
    return "short" if n_turns <= 6 else "medium" if n_turns <= 14 else "long"


def build_eval_set(test_records, n=N_EVAL, seed=SEED):
    """Stratified sample: equal thirds of short, medium and long dialogues."""
    rng = random.Random(seed)
    buckets = {"short": [], "medium": [], "long": []}
    for r in test_records:
        buckets[length_bucket(r["n_turns"])].append(r)
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    out = []
    for k, (name, pool) in zip(per, buckets.items()):
        out += rng.sample(pool, min(k, len(pool)))
    rng.shuffle(out)
    return out


def build_fewshot(train_records, n=N_FEWSHOT, seed=SEED, max_turns=10):
    """Short, clean train dialogues with 2-4 speakers, so demonstrations stay cheap in the prompt."""
    rng = random.Random(seed)
    pool = [r for r in train_records
            if 4 <= r["n_turns"] <= max_turns and 2 <= len(r["speakers"]) <= 4 and "<file" not in r["dialogue"]]
    return rng.sample(pool, n)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def freeze(eval_set, fewshot, overwrite=False):
    """Write the eval and few-shot files once. Refuses to overwrite unless asked."""
    if EVAL_SET_PATH.exists() and not overwrite:
        raise FileExistsError(f"{EVAL_SET_PATH} already exists; the eval set is frozen.")
    save_json(eval_set, EVAL_SET_PATH)
    save_json(fewshot, FEWSHOT_PATH)


def load_eval_set():
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def load_fewshot():
    return json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))


def dataset_summary(records):
    n = len(records)
    return {
        "dialogues": n,
        "avg_turns": round(sum(r["n_turns"] for r in records) / n, 1),
        "avg_dialogue_words": round(sum(r["n_words"] for r in records) / n, 1),
        "avg_summary_words": round(sum(len(r["summary"].split()) for r in records) / n, 1),
        "compression_ratio": round(sum(r["n_words"] for r in records) / sum(len(r["summary"].split()) for r in records), 1),
        "speakers_per_dialogue": dict(sorted(Counter(min(len(r["speakers"]), 5) for r in records).items())),
        "length_buckets": dict(Counter(length_bucket(r["n_turns"]) for r in records)),
        "with_media_tags_pct": round(100 * sum("<file" in r["dialogue"] for r in records) / n, 1),
    }
