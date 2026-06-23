# Phase 2 — bitsandbytes Quantization Findings (#15)

**Goal:** confirm whether bitsandbytes quantization (Q4/Q8) runs on this machine.

**Outcome:** ❌ **It does not.** bitsandbytes quantization requires a CUDA GPU; this
is a CPU-only setup, so AirLLM's `compression='4bit'` fails. This is a documented
result (PLAN ADR-6 flagged the risk in advance).

## What happened

Running `AirLLMRunner(..., compression="4bit").load()` on Qwen2.5-0.5B:

1. AirLLM begins sharding and applies bitsandbytes compression **per layer** (progress
   bar reaches the first of 27 layers).
2. It aborts with:

   ```
   AssertionError: Torch not compiled with CUDA enabled
   ```

The installed torch is the CPU build (`2.12.1+cpu`) and `torch.cuda.is_available()` is
`False` (the laptop's MX350 has only 2 GB and no CUDA-enabled torch build is in use).
bitsandbytes' 4-bit/8-bit quantization kernels are CUDA-only, so the compression step
cannot execute. Recorded in `results/bnb_check_q4_0_5b_*.json`.

## Implication for the experiment

This is a **hardware constraint, not a bug**, and it reshapes the quantization sweep
(#18):

- **FP16** is runnable via AirLLM's CPU layer-sharding (memory-bound, slow — the
  expected behaviour to analyse).
- **Q8 / Q4 (bitsandbytes)** are **not runnable** on this CPU-only machine.

So the FP16-vs-Q8-vs-Q4 comparison cannot be produced here as originally planned. The
honest result is: *quantization via bitsandbytes is gated on CUDA hardware we don't
have.* The report will present this as a finding and, for the quantization dimension,
fall back to a comparison that does not need bitsandbytes (e.g. AirLLM FP16 on-prem vs
the already-quantized Ollama GGUF baseline, which uses llama.cpp CPU kernels rather
than bitsandbytes).

**Reproduce:** `uv run --extra ml python experiments/bitsandbytes_check.py`
