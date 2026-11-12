#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model and save as a standalone model.

Key detail: LoRALinear.fuse() RETURNS a new fused layer (dequantize -> add delta
-> requantize). It does not modify in place, so the traversal must replace the
LoRALinear with the returned layer in its parent module.
"""
import json, os, sys
from mlx_lm import load
from mlx_lm.utils import save
from mlx_lm.tuner.lora import LoRALinear
from mlx.nn import Module

MODEL = "/Users/hermit.dave/llm-models/Qwen3.5-2B-MLX-8bit"
ADAPTER = "/Users/hermit.dave/Projects/ml/distill-qwen35-pilot/work/2b_full_filtered"
DST = "/Users/hermit.dave/llm-models/Qwen3.5-2B-Distilled"

print(f"Loading model + adapter from {ADAPTER}...", flush=True)
model, tokenizer = load(MODEL, adapter_path=ADAPTER)

def fuse_tree(module):
    """Replace every LoRALinear with its fused equivalent, in place."""
    count = 0
    for k in module.keys():
        v = module[k]
        if isinstance(v, LoRALinear):
            module[k] = v.fuse()
            count += 1
        elif isinstance(v, Module):
            count += fuse_tree(v)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, LoRALinear):
                    v[i] = item.fuse()
                    count += 1
                elif isinstance(item, Module):
                    count += fuse_tree(item)
    return count

def count_lora(module):
    n = 0
    for k in module.keys():
        v = module[k]
        if isinstance(v, LoRALinear):
            n += 1
        elif isinstance(v, Module):
            n += count_lora(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, Module):
                    n += count_lora(item)
    return n

print("Fusing LoRA layers...", flush=True)
fused = fuse_tree(model)
print(f"  fused {fused} LoRA layers", flush=True)

remaining = count_lora(model)
print(f"  remaining LoRA layers: {remaining}", flush=True)
if remaining != 0:
    print("ERROR: some LoRA layers were not fused!", file=sys.stderr)
    sys.exit(1)

print(f"Saving merged model to {DST}...", flush=True)
os.makedirs(DST, exist_ok=True)
config = json.load(open(os.path.join(MODEL, "config.json")))
save(DST, MODEL, model, tokenizer, config)

# verify
print("\nVerifying saved model...", flush=True)
size = 0
for f in os.listdir(DST):
    p = os.path.join(DST, f)
    if os.path.isfile(p):
        size += os.path.getsize(p)
print(f"  total size: {size / 1e9:.2f} GB")
print(f"  files: {sorted(os.listdir(DST))}")
print("\nDone.")
