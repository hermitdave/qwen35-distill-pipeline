# Qwen3.5 Distillation Pipeline

A complete, reproducible pipeline for distilling large Qwen teachers into small
Qwen3.5 students on Apple Silicon with MLX. It covers data preparation, LoRA
training, NaN debugging, adapter merging, and benchmarking.

**Verified end-to-end on M3 Max 64 GB** — the 2B student below was trained and
benchmarked with this exact code.

---

## Results

Distilling a mixed Flash Next + Agnes 3 Qwen coding dataset into **Qwen3.5-2B**:

| Category | Base 2B | Distilled 2B | Delta |
|---|---|---|---|
| Syntax & Surface | 100 | 100 | +0 |
| Logic & Algorithmic | 75 | 88 | **+13** |
| Subtle & Tricky | 100 | 88 | −12 |
| Red Herring Resistance | 100 | 100 | +0 |
| Multi-Turn Debugging | 80 | 80 | +0 |
| **Overall** | **91** | **91** | **+0** |

The distilled model is **overall neutral** with a clear trade: algorithmic
problem-solving improves, edge-case fragility appears. Every failure observed was
a *"missing one detail"* bug (a forgotten `import`, a forgotten `.sort()`), not a
reasoning regression.

Training: 5,600 iterations, val loss 0.746 → 0.578, zero NaN after the fix
described below.

---

## Repository structure

```
qwen35-distill-pipeline/
├── README.md                     # this file
├── configs/
│   └── 2b_full_filtered.yaml     # verified training config
├── scripts/
│   ├── build_pilot_data.py       # Flash/Agnes/Alpaca -> chat format
│   ├── filter_dataset.py         # remove over-length prompts (NaN fix)
│   ├── merge_adapter.py          # fuse LoRA into the base model
│   ├── validate_adapter.py       # coding execution + general ability
│   ├── benchmark.py              # leaderboard benchmark (5 categories)
│   └── benchmark_detail.py       # per-task benchmark with failure details
└── docs/
    ├── nan-root-cause.md         # the empty-loss-mask NaN, fully diagnosed
    └── capacity-analysis.md      # why 2B before 4B before 9B
```

---

## Requirements

* macOS with Apple Silicon (tested on M3 Max, 64 GB)
* Python 3.11+ with `mlx`, `mlx_lm`, `transformers`, `pyarrow`
* A local Qwen3.5 student checkpoint in MLX format
* Teacher-generated data in one of the supported formats

The reference environment is the oMLX-bundled Python (`~/.omlx/bin/omlx-cluster-python`),
which ships a working `mlx_lm`.

---

## Usage

### 1. Prepare the data

```bash
python scripts/build_pilot_data.py
```

Converts Flash (`{"text": ...}`), Agnes (`{"prompt", "response"}`), and Alpaca
into the unified `{"messages": [...]}` format that `mlx_lm.lora` expects.

### 2. Filter over-length prompts — **do not skip this**

```bash
python scripts/filter_dataset.py
```

Removes examples whose prompt is ≥ `max_seq_length`. Without this, training will
hit an **empty loss mask** and produce `0/0 = NaN`, poisoning the run. See
[`docs/nan-root-cause.md`](docs/nan-root-cause.md) for the full diagnosis.

### 3. Train

```bash
python -m mlx_lm.lora -c configs/2b_full_filtered.yaml \
    --adapter-path adapters/2b
```

Key settings (see the YAML for the full list):

| Setting | Value | Why |
|---|---|---|
| `mask_prompt` | `true` | Prevents the loss from covering prompt tokens |
| `rank` / `scale` | 16 / 16 | Standard; matches the prior 27B→9B run |
| `num_layers` | 16 | Half the 2B's 24 layers — fits comfortably |
| `learning_rate` | 1e-5 | Stable where 2e-5 diverged |
| `max_seq_length` | 512 | Matches the filtered data |

### 4. Merge the adapter

```bash
python scripts/merge_adapter.py
```

Fuses the LoRA into the base weights and writes a standalone model. Two
non-obvious details are handled for you:

* `LoRALinear.fuse()` **returns** a new layer — it does not modify in place.
* The layer tree stores `layers` as a **list**, so the traversal must recurse
  into lists as well as `nn.Module` children.

### 5. Validate and benchmark

```bash
python scripts/validate_adapter.py <adapter_dir> <out.json>
python scripts/benchmark.py
python scripts/benchmark_detail.py subtle
```

`benchmark.py` reproduces the five-category leaderboard;
`benchmark_detail.py` shows exactly which task failed and the generated code.

---

## Configuration reference

```yaml
model: /path/to/Qwen3.5-2B-MLX-8bit
data: /path/to/filtered/data          # must contain train.jsonl + valid.jsonl
train: true
fine_tune_type: lora
mask_prompt: true
num_layers: 16
batch_size: 2
grad_accumulation_steps: 2
iters: 5600
learning_rate: 0.00001
lora_parameters: {rank: 16, dropout: 0.0, scale: 16.0}
max_seq_length: 512
steps_per_eval: 100
steps_per_report: 50
save_every: 50                        # frequent checkpoints — cheap insurance
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Loss goes NaN at a fixed iteration, regardless of LR/optimizer | Empty loss mask | Run `filter_dataset.py`; see `docs/nan-root-cause.md` |
| Loss oscillates then diverges | LR too high | Lower to 1e-5 |
| `Metal GPU Hang Error` | Transient driver issue | Resume from the last checkpoint |
| Merge reports 0 layers fused | `fuse()` returns a new layer | Use `scripts/merge_adapter.py` |
| Merged model is smaller than the base | Vision tower dropped | Expected — the merge is text-only |

---

## Background

* **[docs/nan-root-cause.md](docs/nan-root-cause.md)** — the empty-loss-mask NaN
  that cost hours, and the two-line check that prevents it.
* **[docs/capacity-analysis.md](docs/capacity-analysis.md)** — why the pilot
  order is 2B → 4B → 9B, and why the 9B is the hardest student despite being the
  original target.

---

## Attribution

This pipeline was developed with **Hermes Agent** (Nous Research), which
performed the data preparation, training, debugging, merging, and benchmarking.

Teacher models: **Qwen3.8-Flash-Next** (Qwen Community License 1.0) and
**Agnes 3 Qwen** (Agnes AI). Student: **Qwen3.5-2B** (Apache 2.0).

The training data is regenerable from those teachers and is **not** redistributed
here — only the pipeline is published, to avoid mixing incompatible licenses.

## License

MIT — see [LICENSE](LICENSE).
