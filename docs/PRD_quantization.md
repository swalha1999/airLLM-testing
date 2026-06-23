# Dedicated PRD — Quantization

**Version:** 1.00
**Parent:** [PRD.md](PRD.md) · **Design:** [PLAN.md](PLAN.md)
**Mechanism:** reducing weight numeric precision to cut memory and I/O, trading some output quality.

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
**bitsandbytes**.

## 2. Requirements

| ID | Requirement |
|---|---|
| R1 | Run the model at ≥2 precision levels (target: FP16, Q8, Q4). |
| R2 | Keep prompt, `max_new_tokens`, and shard path identical across levels (controlled experiment). |
| R3 | Record memory, TTFT, TPOT, throughput, runtime for each level. |
| R4 | Assess output quality per level against a fixed prompt set with a stated rubric. |
| R5 | Identify the quality "red line" and report it. |
| R6 | If a level fails to run (e.g. bitsandbytes on Windows/CPU), document the failure as a result. |

## 3. Expected I/O

- **Input:** model id, precision level (FP16/Q8/Q4), fixed prompt set, `max_new_tokens`.
- **Output:** generated text per level + one results record per (level, prompt) in `results/`,
  plus a qualitative quality score per level.

## 4. Performance Metrics

- Peak RAM per level (expect monotonic decrease FP16 → Q8 → Q4).
- TTFT / TPOT / throughput per level (expect improvement as bits drop, since less I/O).
- Output-quality rubric score per level (expect degradation as bits drop).
- The **memory/speed vs quality** trade-off curve (the deliverable insight).

## 5. Constraints & Limitations

- **bitsandbytes is fragile on Windows/CPU** (built for CUDA GPUs). Q4/Q8 may not run; a
  documented failure is an acceptable result and itself an engineering finding.
- Quality assessment is qualitative (no benchmark harness in scope) — rubric-based, honest.
- Quantization overhead (pack/unpack) can offset I/O savings on CPU; the data will show which dominates.

## 6. Alternatives Considered & Rationale

| Alternative | Why not (here) |
|---|---|
| GGUF + llama.cpp quantization | Excellent, but bypasses AirLLM; we want AirLLM's compression path on-topic. |
| GPTQ/AWQ pre-quantized checkpoints | Adds another download/toolchain; AirLLM `compression` is simpler and directly comparable. |
| FP32 baseline | Doubles memory for no insight; FP16 is the realistic ceiling here. |

**Chosen:** AirLLM `compression` (Q8/Q4 via bitsandbytes), compared against FP16 — keeps every
scenario inside the same harness for apples-to-apples measurement.

## 7. Success Criteria & Test Scenarios

- **SC1:** at least two precision levels measured and compared on identical inputs.
- **SC2:** a clear memory/speed-vs-quality trade-off is shown with numbers + a chart.
- **SC3:** the quality red line is stated and justified.
- **Test scenarios:** (a) smoke-test Q4 on 0.5B confirms the compression path runs;
  (b) FP16 vs Q8 vs Q4 sweep on the main model; (c) graceful, logged handling when a
  precision level errors out (defensive path).
