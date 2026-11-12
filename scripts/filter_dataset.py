#!/usr/bin/env python3
"""Filter dataset to remove examples whose prompt is longer than max_seq_length.

These examples produce an empty loss mask (offset >= truncated_length), which
makes the loss 0/0 = NaN and poisons the whole training run.
"""
import json, os, random
from pathlib import Path
from transformers import AutoTokenizer

random.seed(42)
ROOT = Path("/Users/hermit.dave/Projects/ml/distill-qwen35-pilot")
TOKENIZER = "/Users/hermit.dave/llm-models/Qwen3.5-2B-MLX-8bit"
MAX_SEQ = 512
tok = AutoTokenizer.from_pretrained(TOKENIZER)

def prompt_len(msgs):
    """Length of the prompt (everything before the assistant response)."""
    add_gen = msgs[-1].get("role") == "assistant"
    return len(tok.apply_chat_template(
        msgs[:-1], add_generation_prompt=add_gen, return_dict=False
    ))

def total_len(msgs):
    return len(tok.apply_chat_template(msgs, return_dict=False))

def filter_file(src, dst, max_seq=MAX_SEQ):
    kept, dropped_prompt, dropped_long = [], 0, 0
    with open(src) as f:
        for line in f:
            ex = json.loads(line)
            msgs = ex["messages"]
            pl = prompt_len(msgs)
            tl = total_len(msgs)
            if pl >= max_seq:
                dropped_prompt += 1
                continue
            # also drop examples whose total length is absurd (> 4x max_seq)
            if tl > 4 * max_seq:
                dropped_long += 1
                continue
            kept.append(ex)
    with open(dst, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")
    return len(kept), dropped_prompt, dropped_long

for split in ["train", "valid"]:
    src = ROOT / f"data/full/{split}.jsonl"
    dst = ROOT / f"data/full_filtered/{split}.jsonl"
    os.makedirs(ROOT / "data/full_filtered", exist_ok=True)
    kept, dp, dl = filter_file(src, dst)
    print(f"{split}: kept {kept}, dropped (prompt too long) {dp}, dropped (total too long) {dl}")

# sanity check: no example with prompt >= max_seq
for split in ["train", "valid"]:
    with open(ROOT / f"data/full_filtered/{split}.jsonl") as f:
        lines = f.readlines()
    bad = 0
    for line in lines:
        ex = json.loads(line)
        if prompt_len(ex["messages"]) >= MAX_SEQ:
            bad += 1
    print(f"{split}: {len(lines)} examples, {bad} with prompt >= {MAX_SEQ}")
