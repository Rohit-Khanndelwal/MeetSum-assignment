"""Shared model loading and generation.

Everyone calls generate(); nobody loads their own copy of the model. Importing this module
needs torch + transformers, so the rest of the package imports it lazily.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_ID, SEED

_tok = None
_model = None


def load_model(model_id=MODEL_ID):
    global _tok, _model
    if _model is None or _model.name_or_path != model_id:
        _tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
        _model.eval()
    return _tok, _model


def _encode(messages):
    tok, model = load_model()
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    return tok, model, ids


def generate(messages, max_new_tokens=256, do_sample=False, temperature=1.0, top_p=1.0, top_k=0, seed=SEED):
    """Chat messages in, generated text out.

    Every decoding knob is set explicitly so Qwen's own generation_config defaults
    (temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05) never leak into an experiment.
    top_k=0 disables top-k filtering.
    """
    tok, model, ids = _encode(messages)
    torch.manual_seed(seed)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, repetition_penalty=1.0,
              pad_token_id=tok.eos_token_id)
    if do_sample:
        kw.update(temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        kw.update(temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


# ---- Evidence for the LLM-architecture section (report section 4) ----

def model_summary():
    _, model = load_model()
    c = model.config
    return {
        "model_id": model.name_or_path,
        "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "vocab_size": c.vocab_size,
        "hidden_size": c.hidden_size,
        "num_layers": c.num_hidden_layers,
        "num_attention_heads": c.num_attention_heads,
        "num_kv_heads": getattr(c, "num_key_value_heads", c.num_attention_heads),
        "max_position_embeddings": c.max_position_embeddings,
    }


def tokenize_demo(text):
    tok, _ = load_model()
    ids = tok(text)["input_ids"]
    return {"n_tokens": len(ids), "tokens": tok.convert_ids_to_tokens(ids), "ids": ids}


def token_trace(messages, steps=5, top=3):
    """Greedy-decode a few tokens and show the top-k next-token probabilities at each step."""
    tok, model, ids = _encode(messages)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=steps,
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             repetition_penalty=1.0, output_scores=True,
                             return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
    rows = []
    for step, scores in enumerate(out.scores):
        probs = torch.softmax(scores[0].float(), dim=-1)
        p, idx = probs.topk(top)
        chosen = out.sequences[0, ids.shape[1] + step].item()
        rows.append({
            "step": step + 1,
            "chosen_token": tok.decode([chosen]),
            "top_candidates": [(tok.decode([i]), round(q, 4)) for i, q in zip(idx.tolist(), p.tolist())],
        })
    return rows


def sequence_logprob(messages, target):
    """Exact log p(target | prompt) via the chain rule: sum of log p(token_t | everything before it).

    Only an autoregressive model gives this likelihood in one forward pass (teacher forcing);
    GANs have none and VAEs / diffusion models only give a bound.
    """
    tok, model, ids = _encode(messages)
    tgt = tok(target, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    full = torch.cat([ids, tgt], dim=1)
    with torch.no_grad():
        logits = model(full, attention_mask=torch.ones_like(full)).logits
    # logits at position t predict token t+1, so the target tokens are scored by the positions before them
    logp = torch.log_softmax(logits[0, ids.shape[1] - 1:-1].float(), dim=-1)
    token_logp = logp.gather(1, tgt[0].unsqueeze(1)).squeeze(1)
    total = token_logp.sum().item()
    n = tgt.shape[1]
    return {"n_tokens": n, "logprob": round(total, 2), "avg_logprob": round(total / n, 3),
            "perplexity": round(float(torch.exp(-token_logp.mean())), 2)}
