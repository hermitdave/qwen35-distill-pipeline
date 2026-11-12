#!/usr/bin/env python3
"""Build pilot dataset for 2B distillation.

Sources:
  - Flash data (chat text) from distill-qwen35-9b/flash_data/mlx_format
  - Agnes data (prompt/response) from distill-qwen35-9b/agnes_data
  - Alpaca (general instruction) from alpaca_raw parquet

Output format: {"messages": [{"role":"user",...},{"role":"assistant",...}]}
so mlx_lm.lora can apply the chat template with mask_prompt=True.

Output:
  - pilot/train.jsonl   (1000 examples: 700 coding / 300 general)
  - pilot/valid.jsonl   (100 examples: 70 coding / 30 general)
  - full/train.jsonl    (all coding + general, for later full runs)
"""
import json, os, re, random
import pyarrow.parquet as pq

random.seed(42)

ROOT = "/Users/hermit.dave/Projects/ml/distill-qwen35-pilot"
FLASH_DIR = "/Users/hermit.dave/Projects/ml/distill-qwen35-9b/flash_data/mlx_format"
AGNES_DIR = "/Users/hermit.dave/Projects/ml/distill-qwen35-9b/agnes_data"
ALPACA_PARQUET = os.path.join(ROOT, "data/alpaca_raw/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
OUT = os.path.join(ROOT, "data")

os.makedirs(os.path.join(OUT, "pilot"), exist_ok=True)
os.makedirs(os.path.join(OUT, "full"), exist_ok=True)

# Build think tags from codepoints to avoid literal stripping
THINK_OPEN = ''.join(chr(c) for c in [0x3c,0x74,0x68,0x69,0x6e,0x6b,0x3e])
THINK_CLOSE = ''.join(chr(c) for c in [0x3c,0x2f,0x74,0x68,0x69,0x6e,0x6b,0x3e])
THINK_RE = re.compile(THINK_OPEN + r'.*?' + THINK_CLOSE, re.DOTALL)
THINK_OPEN_RE = re.compile(THINK_OPEN + r'.*', re.DOTALL)

def strip_thinking(text):
    text = THINK_RE.sub('', text)
    text = THINK_OPEN_RE.sub('', text)
    return text.strip()

def parse_flash_text(text):
    """Parse '<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}'."""
    parts = text.split("<|im_start|>")
    if len(parts) < 3:
        return None
    user_part = parts[1]
    assistant_part = parts[2]
    if "\n" not in user_part or "\n" not in assistant_part:
        return None
    prompt = user_part.split("\n", 1)[1].split("<|im_end|>")[0].strip()
    response = strip_thinking(assistant_part.split("\n", 1)[1])
    if not prompt or not response:
        return None
    return prompt, response

def msg(prompt, response):
    return {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]}

# ---------- 1. Flash data ----------
print("Loading Flash data...", flush=True)
flash = []
with open(os.path.join(FLASH_DIR, "train.jsonl")) as f:
    for line in f:
        ex = json.loads(line)
        r = parse_flash_text(ex["text"])
        if r:
            flash.append(msg(*r))
print(f"  Flash: {len(flash)}", flush=True)

# ---------- 2. Agnes data ----------
print("Loading Agnes data...", flush=True)
agnes = []
for fn in ["agnes_all.jsonl", "agnes_phase2.jsonl", "agnes_phase3.jsonl", "agnes_phase4.jsonl"]:
    path = os.path.join(AGNES_DIR, fn)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            prompt = ex["prompt"].strip()
            response = strip_thinking(ex["response"])
            if not prompt or not response:
                continue
            agnes.append(msg(prompt, response))
print(f"  Agnes: {len(agnes)}", flush=True)

# ---------- 3. Alpaca ----------
print("Loading Alpaca...", flush=True)
table = pq.read_table(ALPACA_PARQUET)
alpaca_rows = table.to_pylist()
print(f"  Alpaca raw: {len(alpaca_rows)}", flush=True)
alpaca = []
for ex in alpaca_rows:
    instr = ex.get("instruction", "").strip()
    inp = ex.get("input", "").strip()
    out = ex.get("output", "").strip()
    if not instr or not out:
        continue
    prompt = instr + ("\n\n" + inp if inp else "")
    alpaca.append(msg(prompt, out))
print(f"  Alpaca usable: {len(alpaca)}", flush=True)

# ---------- 4. Sample pilot ----------
CODING_PILOT = 700
GENERAL_PILOT = 300
CODING_VALID = 70
GENERAL_VALID = 30

coding_all = flash + agnes
random.shuffle(coding_all)
random.shuffle(alpaca)

coding_valid = coding_all[:CODING_VALID]
coding_train = coding_all[CODING_VALID:CODING_VALID + CODING_PILOT]
general_valid = alpaca[:GENERAL_VALID]
general_train = alpaca[GENERAL_VALID:GENERAL_VALID + GENERAL_PILOT]

pilot_train = coding_train + general_train
pilot_valid = coding_valid + general_valid
random.shuffle(pilot_train)
random.shuffle(pilot_valid)

print(f"\nPilot train: {len(pilot_train)} (coding {len(coding_train)}, general {len(general_train)})")
print(f"Pilot valid: {len(pilot_valid)} (coding {len(coding_valid)}, general {len(general_valid)})")

def write_jsonl(path, items):
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"  wrote {path} ({len(items)} lines)")

write_jsonl(os.path.join(OUT, "pilot", "train.jsonl"), pilot_train)
write_jsonl(os.path.join(OUT, "pilot", "valid.jsonl"), pilot_valid)

# ---------- 5. Full dataset ----------
coding_full = coding_all[CODING_VALID + CODING_PILOT:]
general_full = alpaca[GENERAL_VALID + GENERAL_PILOT:][:2000]
full_train = coding_full + general_full
random.shuffle(full_train)
print(f"\nFull train: {len(full_train)} (coding {len(coding_full)}, general {len(general_full)})")
write_jsonl(os.path.join(OUT, "full", "train.jsonl"), full_train)

# ---------- 6. Sanity check ----------
print("\n=== SANITY CHECK ===")
for name, items in [("pilot_train", pilot_train), ("pilot_valid", pilot_valid)]:
    sample = items[0]
    msgs = sample["messages"]
    print(f"{name}: n_messages={len(msgs)} roles={[m['role'] for m in msgs]}")
    print(f"  user preview: {msgs[0]['content'][:150]!r}")
    print(f"  assistant preview: {msgs[1]['content'][:150]!r}")

print("\nDone.")
