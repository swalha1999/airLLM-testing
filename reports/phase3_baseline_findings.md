# Phase 3 — Baseline Findings (FR-b, #16)

**Goal:** run the 7B model the "normal" way and characterise the bottleneck, as the
anchor every other scenario is compared against.

**Headline:** the same model + prompt was **~146× slower** when it didn't fit in RAM —
a textbook **memory-bound** bottleneck. It didn't crash; it survived by paging ~15 GB to
disk and grinding for over two hours.

## Two baseline runs

| | Ollama Q4 (fits in RAM) | HF FP16 direct (does **not** fit) |
|---|---|---|
| Engine | Ollama / llama.cpp GGUF (~4-bit) | transformers, full FP16 load |
| Model on disk | ~4.7 GB | ~15 GB |
| **Total time** | **55 s** | **8,070 s ≈ 2 h 14 min** |
| Peak RAM (our process) | 0.05 GB* | **11.7 GB resident** (+ ~15 GB committed to swap) |
| TTFT | 46 s (first-call load) | n/a (single-shot timing) |
| Throughput | 0.79 tok/s | ≈ 0.004 tok/s effective |
| Output | coherent 2-sentence answer | the **same** coherent answer |

\* Ollama runs in its own process, so the in-process `psutil` sampler doesn't see its
memory — for Ollama the timing metrics are the meaningful ones.

Raw data: `results/baseline_ollama_q4_*.json`, `results/baseline_hf_direct_fp16_*.json`.

## Evidence of the bottleneck

Loading the four FP16 checkpoint shards got **progressively slower** as RAM filled and the
OS began swapping:

```
shard 1: 83 s   shard 2: 117 s   shard 3: 162 s   shard 4: 195 s
```

Task Manager during the run (`assets/task_manager_screenshots/`):

- **Memory: 15.5 / 15.7 GB (99%)**, **Available: 238 MB**
- **Disk 0 (C:): 100%** — constant paging to the SSD
- **Committed: 30.6 / 38.4 GB** — ~30 GB committed against 15.7 GB of physical RAM, i.e.
  roughly **half the model lived in the page file**
- **CPU 100%**, **GPU (MX350) 0%** — the work is entirely CPU + memory; the GPU is idle

![Task Manager: RAM 99%, Disk 100% during the FP16 load](../assets/task_manager_screenshots/Screenshot%202026-06-24%20122627.jpg)

## Interpretation (ties to FR-f)

The bottleneck is **memory bandwidth / capacity, not compute**. The 7B in FP16 needs ~15 GB
of weights resident; with only 15.7 GB total RAM there is no room, so every forward pass
forces the OS to page weights between RAM and the SSD. Disk is orders of magnitude slower
than RAM, so throughput collapses. This is the exact problem **AirLLM** addresses (#17): by
loading **one layer at a time** it keeps the working set tiny, trading the same disk I/O for
a controlled, predictable cost instead of uncontrolled thrashing.

**Reproduce:** `uv run python experiments/baseline_ollama.py` and
`uv run --extra ml python experiments/baseline_hf_direct.py` (the latter takes ~2 h).
