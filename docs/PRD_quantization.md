# Dedicated PRD — Quantization

**Version:** 1.10
**Parent:** [PRD.md](PRD.md) · **Design:** [PLAN.md](PLAN.md)
**Mechanism:** reducing weight numeric precision to cut memory and I/O, trading some output quality.

> **Hardware finding (v1.10, #15):** bitsandbytes quantization (AirLLM `compression='4bit'/'8bit'`)
> is **CUDA-only** and **does not run on this CPU-only machine** — it fails with
> `AssertionError: Torch not compiled with CUDA enabled` (torch `2.12.1+cpu`). See
> `reports/phase2_quantization_findings.md` and `results/bnb_check_q4_0_5b_*.json`. This PRD is
> revised accordingly: the runnable quantization comparison is **FP16 (AirLLM, on-prem) vs ~Q4
> (Ollama GGUF / llama.cpp CPU kernels)** rather than an AirLLM/bitsandbytes FP16→Q8→Q4 sweep.

---

## 1. Description & Theoretical Background

A model's weights are numbers. In **FP16** each weight uses 16 bits (2 bytes), so a 7B model
is ~14 GB. **Quantization** stores each weight in fewer bits — **Q8** (8-bit, ~½ the size) or
**Q4** (4-bit, ~¼) — by mapping the original range onto a small set of levels plus per-group
scale/zero-point factors.

Why it matters for this assignment: the bottleneck is **memory and disk I/O**, not arithmetic
(see [PRD_airllm_sharding.md](PRD_airllm_sharding.md)). Smaller weights mean **less data moved
per layer**, so quantization should directly improve TTFT/TPOT and lower peak memory — the
clearest lever we have on a memory-bound system. The trade-off is **accuracy**: fewer bits =
coarser weights = some quality loss, which grows as precision drops. The analysis must locate
the **"red line"** — the precision below which output quality becomes unacceptable.

AirLLM exposes this via its `compression` option (`'4bit'` / `'8bit'`), backed by
**bitsandbytes** — but bitsandbytes needs a CUDA GPU, which this machine lacks (see the
finding above). The CPU-runnable quantization path is therefore **GGUF via llama.cpp**
(Ollama's `qwen2.5:7b`), which quantizes to ~4-bit using CPU kernels instead of bitsandbytes.

## 2. Requirements

| ID | Requirement |
|---|---|
| R1 | Run the model at ≥2 **runnable** precision regimes: FP16 (AirLLM, on-prem) and ~Q4 (Ollama GGUF). |
| R2 | Keep prompt, `max_new_tokens`, and (where applicable) shard path identical across regimes (controlled comparison). |
| R3 | Record memory, TTFT, TPOT, throughput, runtime for each regime. |
| R4 | Assess output quality per regime against a fixed prompt set with a stated rubric. |
| R5 | Identify the quality "red line" and report it. |
| R6 | Document the bitsandbytes Q8/Q4 failure as a result (CUDA-only on this CPU machine) — a documented negative result is required, not optional. |

## 3. Expected I/O

- **Input:** model id, precision regime (FP16 AirLLM / ~Q4 GGUF), fixed prompt set, `max_new_tokens`.
- **Output:** generated text per regime + one results record per (regime, prompt) in `results/`,
  plus a qualitative quality score per regime, plus the recorded bitsandbytes-unavailable finding.

## 4. Performance Metrics

- Peak RAM per regime (expect FP16 AirLLM ≫ Q4 GGUF, since 4-bit weights move ~¼ the data).
- TTFT / TPOT / throughput per regime (expect Q4 GGUF much faster — fewer bits + a native CPU engine).
- Output-quality rubric score per regime (expect some degradation at Q4).
- The **memory/speed vs quality** trade-off (the deliverable insight), with the caveat that FP16
  and Q4 also differ by *engine* (AirLLM disk-sharding vs llama.cpp in-memory), noted in analysis.

## 5. Constraints & Limitations

- **bitsandbytes Q8/Q4 confirmed unavailable here** (#15): it is CUDA-only and aborts with
  `Torch not compiled with CUDA enabled` on this CPU build. This is the headline finding, not a
  mere risk — quantization via AirLLM's `compression` cannot be demonstrated on this hardware.
- The FP16-vs-Q4 comparison crosses **two engines** (AirLLM disk-sharding vs llama.cpp/GGUF),
  so it is not a pure precision-only A/B; the analysis must attribute differences carefully.
- Quality assessment is qualitative (no benchmark harness in scope) — rubric-based, honest.
- Quantization overhead (pack/unpack) can offset I/O savings on CPU; the data will show which dominates.

## 6. Alternatives Considered & Rationale

| Alternative | Why not (here) |
|---|---|
| AirLLM `compression` (Q8/Q4 via bitsandbytes) | **Tried first; fails — CUDA-only (#15).** Documented as the finding, but cannot produce results on this CPU machine. |
| GPTQ/AWQ pre-quantized checkpoints | Also lean on CUDA kernels for fast inference; same hardware wall, extra toolchain. |
| FP32 baseline | Doubles memory for no insight; FP16 is the realistic ceiling here. |

**Chosen:** **FP16 via AirLLM (on-prem)** compared against **~Q4 via Ollama GGUF (llama.cpp CPU
kernels)** — the only quantization path that actually runs on this hardware. The original
bitsandbytes plan is retained as a documented negative result.

## 7. Success Criteria & Test Scenarios

- **SC1:** at least two runnable precision regimes (FP16 AirLLM, ~Q4 GGUF) measured on identical inputs.
- **SC2:** a clear memory/speed-vs-quality trade-off is shown with numbers + a chart.
- **SC3:** the quality red line is stated and justified.
- **SC4:** the bitsandbytes Q8/Q4 CUDA-only failure is recorded as a result (#15).
- **Test scenarios:** (a) bitsandbytes 4-bit attempt → documented CUDA failure (done, #15);
  (b) FP16 AirLLM vs Q4 GGUF on the main model; (c) graceful, logged handling when a
  regime errors out (defensive path).
