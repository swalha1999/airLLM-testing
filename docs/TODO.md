# TODO — Task Breakdown

**Version:** 1.00
**Companion to:** [PRD.md](PRD.md), [PLAN.md](PLAN.md)
**Status legend:** ⬜ not started · 🟡 in progress · ✅ done

Owner for all tasks: Mhmdabad (executed via AI agent).

---

## Phase 0 — Planning  ✅
- ✅ Read assignment + submission guidelines
- ✅ Capture target hardware specs
- ✅ Write `docs/PRD.md`
- ✅ Write `docs/PLAN.md`
- ✅ Write `docs/TODO.md`
- ⬜ **Approve all three docs before development** *(definition of done: user sign-off)*

## Phase 1 — Setup & Downloads  ⬜
*(est. 1.5–3 h, mostly passive download time)*
- ⬜ `uv python install 3.11`; `uv init` project on 3.11 *(DoD: `uv run python --version` → 3.11)*
- ⬜ `uv add` deps: airllm, torch (CPU), transformers, accelerate, bitsandbytes, huggingface_hub, psutil, nvidia-ml-py, matplotlib, pandas, numpy *(DoD: `uv.lock` committed)*
- ⬜ Create `.gitignore` (.env, weights, shards, `__pycache__`, results-cache) + `.env-example` (HF_TOKEN)
- ⬜ Create `config/setup.json` + `config/costs.json` (versioned)
- ⬜ Add HuggingFace token to local `.env` *(DoD: token loads from env, never in code)*
- ⬜ Free disk space if needed; create dedicated `layer_shards_saving_path` folder *(DoD: ≥30 GB free)*
- ⬜ Pull baseline Ollama model (`qwen2.5:7b`) *(DoD: `ollama run` responds)*
- ⬜ Download smoke-test model (Qwen2.5-0.5B) *(DoD: present in HF cache)*

## Phase 2 — Pipeline Smoke-Test  ⬜
*(validate before scaling — the "start small" rule)*
- ⬜ Build SDK skeleton: `run_scenario`, `measure`, `save_result`
- ⬜ Build measurement module: TTFT, ITL/TPOT, throughput, RAM/VRAM samplers
- ⬜ Run AirLLM on Qwen2.5-0.5B at Q2/Q4 to confirm sharding + quant work *(DoD: a result JSON written)*
- ⬜ Confirm bitsandbytes quantization actually runs on this machine *(DoD: Q4 run succeeds OR failure documented)*

## Phase 3 — Experiments  ⬜
*(est. 3–5 h compute)*
- ⬜ **FR-b Baseline:** direct run of main model (Ollama, + HF direct attempt). Capture failure/slowness + bottleneck evidence (RAM pressure, swap)
- ⬜ **FR-c AirLLM main run:** Qwen2.5-7B through AirLLM, layer-sharded
- ⬜ **FR-c Quant sweep:** FP16 vs Q8 vs Q4 (whatever runs); record all
- ⬜ **FR-d Metrics:** for every scenario capture TTFT, ITL/TPOT, throughput, peak RAM/VRAM, runtime, est. power
- ⬜ **FR-d Quality:** qualitative output assessment per quant level on fixed prompts
- ⬜ Save ALL raw numbers to `results/` *(DoD: re-runnable graphs)*

## Phase 4 — Analysis & Figures  ⬜
*(est. 1–1.5 h)*
- ⬜ Comparison tables + bar/line charts (TTFT, throughput, memory) → `figures/`
- ⬜ **FR-e Economic analysis:** API cost vs On-Prem (CAPEX+OPEX); break-even vs volume graph; state all assumptions; reasoned recommendation. *(optional: Cloud-GPU third line)*
- ⬜ **FR-f Concept analysis:** map results to Prefill/Decode, mem-bound vs compute-bound, paging analogy
- ⬜ Optional: Roofline-style visualization
- ⬜ **FR-g Extension:** implement + document ≥1 original extension

## Phase 5 — Report (README) & Submission  ⬜
*(est. 1–1.5 h)*
- ⬜ Write deep-dive technical report as `README.md` (hardware, method, baseline, AirLLM, quant, metrics, economics, concept analysis, extension, run instructions)
- ⬜ Embed ALL graphs/tables/screenshots inline in README
- ⬜ Add reproducible run instructions
- ⬜ Final pass: files ≤150 LOC, no secrets, `.env-example` present, `uv.lock` committed
- ⬜ Push to GitHub (author = user only, no co-author trailer)

---

### Definition of Done (project)
All 8 acceptance criteria in [PRD.md](PRD.md) §3 met; README is a self-contained,
reproducible technical report with every required analysis and all figures embedded.
