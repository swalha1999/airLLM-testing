# Phase 3 — Quantization Comparison (FR-c, #18)

**Goal:** quantify what lowering precision buys — memory, speed, quality — using the
quantization path that actually runs on this CPU machine.

**Headline:** ~Q4 is **~3.2× smaller on disk** and **~79× faster** than FP16 here, with
output quality holding up. But the comparison crosses two engines (AirLLM disk-sharding vs
llama.cpp), and bitsandbytes Q4/Q8 — the "pure" AirLLM quantization — **cannot run** (#15).

## Why this isn't the original FP16→Q8→Q4 sweep

The plan was to quantize *the same AirLLM model* to Q8/Q4 via bitsandbytes. That is
**impossible on this hardware**: bitsandbytes kernels are CUDA-only and abort with
`Torch not compiled with CUDA enabled` (recorded in `results/bnb_check_q4_0_5b_*.json`,
write-up in `reports/phase2_quantization_findings.md`). The runnable substitute is the GGUF
path — Ollama's `qwen2.5:7b` quantizes to ~4-bit with llama.cpp CPU kernels.

## The comparison (from saved results, via `experiments/compare_quantization.py`)

| Regime | Precision | Engine | Disk | Throughput | Peak RAM |
|---|---|---|---|---|---|
| `airllm_7b_fp16` | FP16 | AirLLM (layer-sharded, CPU) | **15 GB** | **0.010 tok/s** | 7.74 GB |
| `baseline_ollama_q4` | ~Q4 | Ollama / llama.cpp GGUF | **4.7 GB** | **0.794 tok/s** | (separate proc) |

- **Disk / memory:** Q4 is **~3.2× smaller** (4.7 vs 15 GB). Fewer bits = less data to move
  per layer, the core quantization win on a memory-bound system.
- **Speed:** Q4 is **~79× faster** (0.79 vs 0.010 tok/s).
- **Quality:** both produce coherent answers to the same prompt — no obvious degradation at
  this depth (a fuller quality rubric is #20).

Consolidated record: `results/quantization_comparison_*.json`.

## Honest caveat — two effects are entangled

This is **not** a pure precision-only A/B. The 79× speed gap comes from **two** things at once:

1. **Precision** — 4-bit weights move ~¼ the data of FP16.
2. **Engine** — llama.cpp runs the whole quantized model *in memory* with optimised CPU
   kernels, while AirLLM streams FP16 layers *from disk* one at a time.

Most of the 79× is the **engine/IO difference**, not precision alone. We cannot isolate
precision because the one tool that would (AirLLM + bitsandbytes, same engine, FP16 vs Q4)
does not run here. The report states this rather than over-claiming a clean precision result.

## Takeaway

Quantization is the **highest-leverage knob** on this memory-bound machine: a ~4-bit GGUF
model both **fits comfortably** (4.7 GB, no paging) and runs **~79× faster** than streaming
FP16 from disk. The catch is that the only CPU-viable quantization route is GGUF/llama.cpp —
the GPU-oriented bitsandbytes path the assignment first imagines is closed on this hardware.

**Reproduce:** `uv run python experiments/compare_quantization.py`
