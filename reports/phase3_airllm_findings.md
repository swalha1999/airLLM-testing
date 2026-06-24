# Phase 3 — AirLLM Main Run Findings (FR-c, #17)

**Goal:** run the *same* 7B that thrashed in the baseline (#16), but through AirLLM's
layer-by-layer execution, and show what changes.

**Headline:** AirLLM makes the FP16 model **runnable with low, stable memory and zero
swap** — but it is **~79× slower than the quantized Ollama run**. Feasibility bought with
latency: exactly the paging trade-off the assignment is about.

## Result

| Scenario | Fits in RAM? | Peak RAM | Disk | Speed | Total |
|---|---|---|---|---|---|
| Ollama Q4 (optimised) | ✅ | — (separate proc) | low | **0.79 tok/s** | 55 s |
| **AirLLM FP16 (layer-sharded)** | ✅ **made to fit** | **7.74 GB, no swap** | 21% | **0.01 tok/s** (~100 s/tok) | 797 s / 8 tok |
| HF FP16 direct (#16) | ❌ | 11.7 GB **+ ~15 GB swap** | **100%** | ~0.004 tok/s | 2 h 14 min |

Output (AirLLM): *"…A transformer model is a type of neural"* — coherent.
Raw data: `results/airllm_7b_fp16_*.json`.

## The memory contrast (the whole point)

Task Manager during each FP16 run:

| | Baseline FP16 direct (#16) | AirLLM FP16 (this run) |
|---|---|---|
| Memory in use | **15.5 / 15.7 GB (99%)** | **7.8 / 15.7 GB (50%)** |
| Available | 238 MB | **8.0 GB** |
| Disk (C:) | **100%** (constant paging) | **21%** (steady) |
| Committed | 30.6 / 38.4 GB (≈15 GB in page file) | 13.2 / 36.8 GB (no swap) |

![AirLLM: RAM 50%, Disk 21%, 8 GB free — no swap](../assets/task_manager_screenshots_after/Screenshot%202026-06-24%20143851.jpg)

Same model, same precision — AirLLM holds **half the memory of the naive load and never
touches swap**, because only one (prefetched: ~two) transformer layer is resident at a time
instead of all 28 + embeddings.

## The speed price

AirLLM re-reads the whole model from disk for **every token** — 28 layer files per token.
At ~100 s/token it is ~79× slower than Ollama's in-memory Q4 run. This is not a defect; it
is the cost of trading memory for feasibility. The bottleneck is **disk I/O bandwidth**
(memory-bound, FR-f), not compute — the CPU sat at ~33%.

## Dependency note (another AirLLM/transformers version wall)

The first attempt (transformers 4.48.3) sharded fine but **crashed during generation** with
`TypeError: cannot unpack non-iterable NoneType object` at `cos, sin = position_embeddings`.
transformers ≥ 4.43 moved rotary-position-embedding (RoPE) computation up to the model level
and passes it into each decoder layer; AirLLM 2.11's manual per-layer forward doesn't supply
it, so the layer receives `None`. Pinning **transformers < 4.43** (4.42.4) restores the
pre-refactor behaviour and generation succeeds. (`results/` keeps both the failed and the
successful records.) This narrows the working window to `optimum<2` **and**
`transformers>=4.40,<4.43`.

**Reproduce:** `uv run --extra ml python experiments/airllm_7b.py` (reuses shards in
`C:\airllm_shards` if present).
