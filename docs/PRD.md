# PRD — EX05: Running a Massive LLM Locally (AirLLM, Quantization & Benchmarking)

**Version:** 1.00
**Owner:** Mhmdabad
**Course:** L08 — Assignment 05 (Dr. Yoram Segal)
**Status:** Draft — pending approval before development

---

## 1. Overview & Context

This project is a **deep-dive engineering experiment**, not a product. The goal is to
prove — in a measured, reasoned way — that we understand the full pipeline of running a
**deliberately oversized Large Language Model on local hardware** (On-Premises), the
bottlenecks that arise, and the techniques (**AirLLM layer-sharding + quantization**) that
make such a model runnable on a constrained machine.

The central artifact is a **deep-dive technical report** (delivered as the repo `README.md`)
that documents the experiment, embeds all graphs/tables/screenshots inline, and explains
*why* each result happens in terms of LLM inference theory (Prefill vs Decode,
memory-bound vs compute-bound, virtual memory / paging).

### Target hardware (the system under test)
| Resource | Value |
|---|---|
| CPU | Intel i7-1195G7 — 4 cores / 8 threads |
| RAM | 15.7 GB |
| GPU | NVIDIA MX350 (2 GB VRAM) + Intel Iris Xe (iGPU) |
| Disk (free) | ~44 GB on C: |
| OS | Windows 11 |

This is intentionally a **memory-constrained laptop** — ideal for demonstrating the
"model too big for RAM" bottleneck the assignment is built around.

## 2. Problem Statement

A modern LLM in full precision needs far more memory than a typical laptop has. Loading
such a model directly either fails with an out-of-memory error or swaps to disk and becomes
unusably slow. The user problem: *how do you run a model that does not fit, and is it ever
worth doing locally instead of calling a hosted API?*

## 3. Goals, Success Metrics & Acceptance Criteria

### Goals
- G1 — Demonstrate and **identify the real bottleneck** of a direct local run (RAM/VRAM vs compute).
- G2 — Use **AirLLM + quantization** to make an otherwise-unrunnable model run, and explain how.
- G3 — **Measure** standardized inference metrics rigorously and present them visually.
- G4 — Produce a **break-even economic analysis** (On-Prem vs API) as a function of usage.
- G5 — Tie every result back to **inference theory**, not just raw numbers.

### KPIs / Acceptance Criteria
| # | Criterion | Done when |
|---|---|---|
| AC1 | Hardware fully documented & model choice justified | README §hardware complete |
| AC2 | Baseline run attempted & its bottleneck characterized | Baseline result (even if failure) documented |
| AC3 | AirLLM pipeline runs a model with ≥2 quant levels compared | FP16/Q8/Q4 table populated |
| AC4 | All core metrics measured & graphed | TTFT, ITL/TPOT, throughput, peak RAM/VRAM, runtime, power |
| AC5 | Break-even graph (cost vs volume) with all assumptions stated | Graph + recommendation in README |
| AC6 | Concept analysis links results to Prefill/Decode, mem-bound/compute-bound | Narrative section complete |
| AC7 | ≥1 original extension implemented & documented | Extension section complete |
| AC8 | Reproducible: clear run instructions, raw numbers saved | Anyone can re-run from README |

> **A negative result is acceptable** (e.g. we could not speed the model up) **if it is
> well analyzed.** Output quality of the model is *not* graded; engineering reasoning is.

## 4. Functional Requirements (the 7 Core Tasks)

- **FR-a — Hardware docs & model choice.** Exact CPU/cores/RAM/GPU/VRAM/storage; pick a HF
  model too big to fit comfortably in RAM; justify (params, format, size).
- **FR-b — Baseline (direct run).** Run the model directly (Ollama / HF). Document what
  happens: OOM, stuck on load, or unbearably slow. Identify & explain the bottleneck. This
  is the comparison anchor.
- **FR-c — AirLLM + quantization.** Pipe the model through AirLLM. Test multiple quant
  levels (e.g. FP16, Q8, Q4). Show how layer-sharding changes resource allocation.
- **FR-d — Measure & compare.** Per scenario, measure and tabulate/graph:
  TTFT, ITL/TPOT, throughput (tok/s), peak RAM & VRAM, total runtime, estimated power,
  and a qualitative output-quality assessment per quant level.
- **FR-e — Economic analysis.** Compute API cost (input+output tokens × price) vs On-Prem
  cost (CAPEX amortized + OPEX electricity/maintenance). Derive **break-even** as a
  function of usage volume; plot cumulative-cost-vs-volume; give a reasoned recommendation.
  Optional: add a Cloud-GPU rental third line.
- **FR-f — Concept analysis.** Link results to Prefill vs Decode, VRAM's role,
  memory-bound vs compute-bound, and the virtual-memory/paging analogy AirLLM relies on.
- **FR-g — Original extension.** At least one: new experiment, new metric, interesting
  comparison graph, LoRA/QLoRA, or multi-model size comparison.

## 5. Non-Functional Requirements

**Compliance target: full adherence to Submission Guidelines V3.** The enforced coding
standards (file size, SDK, gatekeeper, versioning, TDD/coverage, linting, package layout,
etc.) are specified once in [PLAN.md](PLAN.md) §2 and apply to all code. Project-level NFRs:

- **NFR1 — Reproducibility:** all raw numbers saved to `results/`; graphs regenerated from them.
- **NFR2 — Tooling:** `uv` only (no pip/venv/python -m); `uv.lock` committed.
- **NFR3 — Security:** no HuggingFace token or secrets in code; `.env` (gitignored) + `.env-example`.
- **NFR4 — Code quality:** files ≤150 LOC, docstrings, descriptive names, no duplication, SDK layer.
- **NFR5 — Presentation:** README is self-contained; all figures/tables embedded inline.
- **NFR6 — Honesty:** report failures faithfully; document negative results with analysis.
- **NFR7 — Testing:** TDD; global test coverage ≥85% (suite fails below); external deps mocked.
- **NFR8 — Linting:** zero Ruff violations across the project.
- **NFR9 — Versioning:** code + every config versioned from 1.00; config version checked at startup.

## 6. Research Questions (must be answered in the report)

1. What was the bottleneck of the direct run (RAM/VRAM vs compute)? How was it identified?
2. How does AirLLM change resource allocation, and how does it relate to virtual memory/paging?
3. What was quantization's effect on memory, speed, paging, and output quality — and where is the "red line" of acceptable quality?
4. How do Prefill and Decode surface in TTFT vs TPOT measurements?
5. What is the price (latency/throughput) of running a big model on modest hardware?
6. When is local economically worthwhile vs an external API?

## 7. Assumptions, Dependencies, Constraints, Out-of-Scope

**Assumptions:** model weights fit on disk after accounting for AirLLM's second (sharded) copy.
**Dependencies:** AirLLM, PyTorch (CPU), transformers/accelerate, bitsandbytes, huggingface_hub, Ollama (baseline), matplotlib/pandas, HuggingFace account+token.
**Constraints:**
- Disk free ~44 GB → caps practical model size at ~7B (download + shards).
- 2 GB VRAM → runs are effectively CPU / memory-bound (reinforces the analysis).
- Python must be 3.12 (3.14 has no torch/bitsandbytes wheels).
- bitsandbytes on Windows/CPU is fragile — failure is an acceptable, documented result.

**Out-of-scope:** training from scratch; serving infrastructure; multi-node; chasing model
output accuracy; running for hours just to inflate numbers.

## 8. Deliverables

A complete public GitHub repo containing: full experiment/measurement code, baseline +
AirLLM + quantization experiments, the deep-dive technical report **as the README** with all
graphs/tables/screenshots embedded, comparison figures, the economic break-even analysis,
the concept-linking analysis, and the documented original extension(s).

## 9. Timeline & Milestones

| Milestone | Deliverable | Est. |
|---|---|---|
| M0 — Planning | PRD / PLAN / TODO approved | — |
| M1 — Setup | Env + models downloaded, pipeline smoke-test on tiny model | 1.5–3 h (mostly downloads) |
| M2 — Experiments | Baseline + AirLLM + quant runs, raw metrics captured | 3–5 h compute |
| M3 — Analysis | Data processed, graphs + break-even built | 1–1.5 h |
| M4 — Report | README technical report written | 1–1.5 h |

Active "screen" work ≈ 2–3 h; remainder is downloads + physical compute.
