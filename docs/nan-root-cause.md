# The empty-loss-mask NaN

A training run that goes NaN at a **fixed iteration regardless of hyperparameters**
is not a learning-rate problem. It is a data problem — specifically, a batch
whose loss mask is empty.

## Symptom

The loss is healthy for hundreds of iterations, then becomes `nan` and stays
there. Changing the learning rate, optimizer, LoRA scale, or even the training
framework does **not** move the failure point.

Measured on Qwen3.5-2B:

| Config | NaN reported at |
|---|---|
| adam, lr 1e-5, scale 16 | iteration 350 |
| adam, lr 5e-6, scale 8 | iteration 350 |
| sgd, lr 1e-3, scale 16 | iteration 350 |
| mlx-lm-lora + fast_vjp, lr 1e-5 | iteration 150 |

The fact that the failure lands on the **same iteration** across different
hyperparameters is the signature: the batch order is deterministic (seeded), and
one specific batch is the trigger.

## Root cause

`mlx_lm.lora` computes the loss with a mask over the response tokens:

```python
mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
ce = nn.losses.cross_entropy(logits, targets) * mask
ntoks = mask.sum()
ce = ce.astype(mx.float32).sum() / ntoks
```

`lengths[:, 0]` is the **prompt length** (the offset) and `lengths[:, 1]` is the
**truncated sequence length**. The mask covers `offset .. truncated_length`.

When `mask_prompt: true` (or the `messages` format is used) and an example's
prompt is longer than `max_seq_length`, the truncation leaves no room for any
response tokens:

```
offset = 4786, max_seq = 512, truncated_length = 512
mask covers steps 4786 .. 512  ->  EMPTY
```

If **both** examples in the batch are like this, `ntoks == 0` and:

```
ce = 0 / 0 = NaN
```

The NaN then poisons the optimizer state, and every subsequent checkpoint is
corrupted. Because the trainer reports the loss every `steps_per_report`
iterations, the NaN appears at the **next report boundary** after the bad batch
(e.g. a bad batch at iteration 324 shows up as NaN at iteration 350).

## Detection

Reproduce the trainer's batch order and check for empty masks. This takes
seconds:

```python
import numpy as np
from mlx_lm.tuner.datasets import load_local_dataset, CacheDataset
from mlx_lm.tuner.trainer import iterate_batches

train, valid, test = load_local_dataset(data_path, tokenizer, args)
cd = CacheDataset(train)
np.random.seed(0)   # must match the training seed
it = iterate_batches(dataset=cd, batch_size=2, max_seq_length=512, loop=True)

for i, (batch, lengths) in enumerate(it, 1):
    info = lengths.tolist()
    total = sum(max(0, ln - off) for off, ln in info)
    if total == 0:
        print(f"EMPTY MASK at iteration {i}: {info}")
```

## Fix

Filter the dataset to drop examples whose prompt is at least `max_seq_length`:

```python
def prompt_len(messages):
    add_gen = messages[-1].get("role") == "assistant"
    return len(tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=add_gen, return_dict=False))

if prompt_len(messages) >= max_seq_length:
    continue   # drop this example
```

`scripts/filter_dataset.py` in this repository does exactly this.

## Why it matters

This bug is silent until it isn't. The first NaN is reported dozens of
iterations after the offending batch, so the obvious suspect (the current
learning rate) is the wrong one. Filtering takes seconds and saves hours of
lost compute — run it before every long training job.
