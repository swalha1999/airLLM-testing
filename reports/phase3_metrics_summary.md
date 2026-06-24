# Phase 3 — Consolidated Metrics (FR-d, #19)

**Goal:** line up the standardized inference metrics for every scenario in one place.

Built from the saved result records by `experiments/consolidate_metrics.py` (recorded to
`results/metrics_summary_*.json`), so it regenerates from data — no re-running (NFR1).

## Consolidated table

| Scenario | Peak RAM (GB) | Peak VRAM (GB) | TTFT (s) | TPOT (ms) | Throughput (tok/s) | Runtime (s) | Power (Wh) |
|---|---|---|---|---|---|---|---|
| `baseline_ollama_q4` | 0.05* | 0.43 | 46.3 | 206.0 | **0.794** | 55.4 | 0.69 |
| `baseline_hf_direct_fp16` | **11.73** | 0.43 | N/A | N/A | N/A | **8070.6** | N/A |
| `airllm_7b_fp16` | 7.74 | 0.09 | N/A | N/A | **0.010** | 797.6 | N/A |

\* Ollama runs in its own process, so the in-process `psutil` sampler doesn't capture its
RAM — the model is ~4.7 GB GGUF. The other VRAM figures are background GPU usage (the runs
are CPU-only; the MX350 sits idle).

## Why some cells are N/A (honest measurement notes)

The metric set is captured per scenario by whichever harness path the run used:

- **`baseline_ollama_q4`** went through the `StreamMeasurer` (token stream over Ollama's HTTP
  API), so it has the full set: TTFT, TPOT, throughput. Its TTFT (46 s) is dominated by the
  first-call model load.
- **`baseline_hf_direct_fp16`** and **`airllm_7b_fp16`** used a `ResourceSampler` + total-time
  wrapper (not a token stream), because the priority there was **peak RAM** and total runtime —
  the bottleneck/feasibility evidence. They therefore record peak RAM + runtime (and AirLLM a
  derived throughput) but not per-token TTFT/TPOT. AirLLM's per-layer execution also doesn't
  expose a clean streaming hook for TTFT without extra instrumentation.

This is a deliberate, documented trade-off, not missing data: each scenario captured the
metrics that answer its question (timing for the clean run; memory + runtime for the
bottleneck and feasibility runs).

## Headline numbers (the story in three rows)

- **Ollama Q4** (fits): 0.79 tok/s, 55 s — the well-behaved reference.
- **HF FP16 direct** (doesn't fit): 11.7 GB resident + ~15 GB swap, **8,070 s** — the
  memory-bound bottleneck.
- **AirLLM FP16** (made to fit): 7.74 GB **no swap**, 0.010 tok/s — feasibility at ~79× the
  latency of Q4.

**Reproduce:** `uv run python experiments/consolidate_metrics.py`
