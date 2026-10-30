"""Project-wide constants. Change a value here, never inside a notebook."""
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # use if Colab GPU quota runs out

DATASET_ID = "knkarthick/samsum"   # parquet copy of SAMSum: id, dialogue, summary

SEED = 42
N_EVAL = 50            # test-split dialogues in the frozen eval set
N_FEWSHOT = 4          # train-split demonstrations (never overlap the test split)
CONCISE_MAX_WORDS = 40

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"
FEWSHOT_PATH = DATA_DIR / "fewshot.json"
LOG_PATH = LOG_DIR / "experiment_log.csv"
