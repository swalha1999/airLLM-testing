# Phase 3 — Output-Quality Assessment (FR-d, #20)

**Goal:** judge the output quality of each scenario and look for the precision "red line"
where quality becomes unacceptable.

**Headline:** across every regime that *ran* (FP16 and ~Q4), output quality is **good and
indistinguishable** — no visible degradation. The red line was **not reached**, because the
only sub-Q4 path (bitsandbytes Q8/Q4) is CUDA-gated and could not run here (#15).

## Rubric

Qualitative, 1–5 (5 = best), on a fixed prompt ("Explain what a transformer model is in two
sentences"):

- **Coherence** — does it read as sensible, well-formed text?
- **Relevance** — is it on-topic and correct?
- **Fluency** — grammar / naturalness.

Scores are a human judgement; they are stated here and recorded in
`results/quality_assessment_*.json` (built by `experiments/assess_quality.py`) so they are
transparent and tied to the captured outputs.

## Assessment

| Scenario | Precision | Coherence | Relevance | Fluency | Note |
|---|---|---|---|---|---|
| `baseline_ollama_q4` | ~Q4 (GGUF) | 5 | 5 | 5 | Complete, accurate, fluent — no visible quantization loss |
| `baseline_hf_direct_fp16` | FP16 | 5 | 5 | 5 | Complete, accurate, fluent (cut at the 32-token cap) |
| `airllm_7b_fp16` | FP16 | 5 | 5 | 4 | On-topic and correct, but only 8 tokens (cap) so partial |

Captured outputs (verbatim, from `results/`):

- **Ollama ~Q4:** *"A transformer model is a type of neural network architecture widely used
  in natural language processing for tasks like translation and text generation, known for
  its mechanism based on self-attention…"*
- **HF FP16 direct:** *"A transformer model is a type of neural network architecture that is
  used for natural language processing tasks such as machine translation, text
  summarization, and language generation. It…"*
- **AirLLM FP16:** *"…A transformer model is a type of neural"* (8-token cap, partial).

## The "red line" (FR-c / PRD_quantization)

The assignment asks where quality falls off as precision drops. On this hardware we could
only test **two** points — FP16 and ~Q4 — and **both look equally good**. We could not push
to Q2 / Q4-bitsandbytes / lower because those need CUDA (#15). So:

> **The quality red line was not crossed in the testable range.** ~Q4 (GGUF) already shows no
> degradation versus FP16 for this task, and we have no CPU-viable route to a precision low
> enough to break it.

This is itself a finding: on this CPU machine the *useful* trade-off is "~Q4 vs FP16", and
quantization to ~Q4 is effectively free in quality terms while paying back ~3.2× memory and
~79× speed (#18) — so the practical recommendation is to quantize.

**Reproduce:** `uv run python experiments/assess_quality.py`
