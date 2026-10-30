# MeetSum

Structured meeting and conversation summarization on SAMSum (dataset option D3), for PCAM ZG516 Applied Generative AI, Group Major Project 1 (Group 21).

MeetSum turns a chat or meeting transcript into a validated JSON record: a summary, participants, key points, decisions and action items. Each action item is then logged through a simulated task-tracker tool.

```json
{
  "summary": "Gene will pick up Felicity at 7.30 to watch the Oscar-nominated Polish movie Cold War at 8.",
  "participants": ["Felicity", "Gene"],
  "key_points": ["Felicity suggests the Polish movie Cold War, nominated for Oscars"],
  "decisions": ["They will see the 8 pm show tonight"],
  "action_items": [{"owner": "Gene", "task": "Pick up Felicity", "due": "tonight at 7:30"}]
}
```

## Run it on Colab

1. Upload this whole `MeetSum` folder to Google Drive as `MyDrive/MeetSum`.
2. Open `00_master_notebook.ipynb` in Colab and set Runtime → Change runtime type → **T4 GPU**.
3. Run the cells top to bottom. Each section names its owner (M1–M4) and the requirement or experiment it covers.

## How it maps to the assignment

| Requirement / experiment | Where | Owner |
| --- | --- | --- |
| Dataset understanding | Notebook §1, `meetsum/data.py` | M2 |
| Generative-model families compared (VAE, GAN, flow, diffusion, causal LLM) and why a causal LLM | Notebook §2.0, `meetsum/llm.py` (`sequence_logprob`) | M2 |
| LLM architecture | Notebook §2.1, `meetsum/llm.py` (`model_summary`, `tokenize_demo`, `token_trace`) | M2 |
| E1 baseline | Notebook §3, prompt P0 | M2 |
| E2 prompt comparison, concise vs detailed | Notebook §4, `meetsum/prompts.py` (P0–P4, `detail` parameter) | M3 |
| E3 decoding (greedy, temperature, top-p, top-k) | Notebook §5, `experiment.DECODING`, self-ROUGE-L for variability | M3 |
| Programmatic prompting | Notebook §6, `PromptTemplate` | M3 |
| E4 structured output | Notebook §7, `meetsum/schema.py` | M4 |
| E5 tool calling | Notebook §8, `meetsum/tools.py` (`fetch_transcript`, `create_task`) | M4 |
| Adaptation: prompting vs RAG vs PEFT | Notebook §9: zero-shot vs fixed few-shot vs BM25-retrieved few-shot | M4 |
| E7 final system | Notebook §10, `meetsum/pipeline.py` (`summarize_meeting`) | M1 |
| E6 failure analysis | Notebook §11, `metrics.categorize_failure` | M3 |
| Responsible use | Notebook §12, `meetsum/safety.py` | M4 + M1 |

## Evaluation criteria

| Criterion | What it measures |
| --- | --- |
| ROUGE-1 / ROUGE-2 / ROUGE-L | Overlap with the human reference summary |
| Length compliance | % of summaries within 40 words (the concise target) |
| Hallucinated-name rate | % of summaries naming a person, place or thing that never appears in the chat |
| Speaker coverage | Share of speakers the summary mentions |
| JSON validity, grounding rate | Valid schema; participants are real speakers; action-item owners are named in the chat |

## Data

- `data/eval_set.json`: **frozen**. 50 test dialogues, seed 42, stratified 17 short (≤ 6 turns) / 17 medium (7–14) / 16 long (15+).
- `data/fewshot.json`: 4 train dialogues. Each has a **hand-written structured target** (decisions and action items), because SAMSum only provides free-text summaries.
- Never rebuild or edit either file. If they change, results stop being comparable across members.

## Facts from the data (checked locally)

- SAMSum test split: 819 dialogues, 11.2 turns and 95.5 words on average. Reference summaries average 20 words, a 4.8× compression ratio.
- 76% of test dialogues have exactly 2 speakers, and 16% contain media tags such as `<file_gif>`.
- The hallucinated-name check flags **4.4% of the human reference summaries** (spelling variants, places like "New York"). Treat that as the metric's noise floor when reading model results.

## Rules for the team

- Use `meetsum.llm.generate()`. Never load your own model copy.
- Set `OWNER` in the setup cell before logging runs. Every experiment appends to `logs/experiment_log.csv`.
- Change shared code in `meetsum/`, not inside notebook cells. Run the offline tests before you share a change:

```bash
pip install pydantic rank_bm25 rouge_score datasets
python tests/test_offline.py
```
