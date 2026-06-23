# TODO — Task Breakdown

**Version:** 1.00
**Companion to:** [PRD.md](PRD.md), [PLAN.md](PLAN.md)
**Status legend:** ⬜ not started · 🟡 in progress · ✅ done

Owner for all tasks: Mhmdabad (executed via AI agent).

---

> **Standing rule (V3 full compliance):** every code module is written TDD-style with its
> own test file (happy + error path), stays ≤150 lines, passes Ruff clean, and routes
> external calls through the gatekeeper. See [PLAN.md](PLAN.md) §2 for all enforced standards.

## Phase 0 — Planning  🟡
- ✅ Read assignment + submission guidelines
- ✅ Capture target hardware specs
- ✅ Write `docs/PRD.md`
- ✅ Write `docs/PLAN.md`
- ✅ Write `docs/TODO.md`
- ✅ Add consolidated Coding Standards section (PLAN §2); set full-V3-compliance target
- ✅ Write dedicated PRDs: `PRD_airllm_sharding.md`, `PRD_quantization.md`, `PRD_cost_model.md`
- ⬜ **Approve all docs before development** *(definition of done: user sign-off)*

## Phase 1 — Setup & Downloads  ⬜
*(est. 1.5–3 h, mostly passive download time)*
- ⬜ `uv python install 3.11`; `uv init` project on 3.11 *(DoD: `uv run python --version` → 3.11)*
- ⬜ `uv add` deps: airllm, torch (CPU), transformers, accelerate, bitsandbytes, huggingface_hub, psutil, nvidia-ml-py, matplotlib, pandas, numpy *(DoD: `uv.lock` committed)*
- ⬜ `uv add --dev` ruff, pytest, pytest-cov
- ⬜ Configure `pyproject.toml`: ruff (`E,F,W,I,N,UP,B,C4,SIM`) + coverage `fail_under = 85`
- ⬜ Create package skeleton: `src/<pkg>/__init__.py` (+`__all__`, `__version__`), `shared/version.py` = "1.00"
- ⬜ Create `.gitignore` (.env, weights, shards, `__pycache__`, results-cache) + `.env-example` (HF_TOKEN)
- ⬜ Create `config/setup.json` + `config/costs.json` + `config/rate_limits.json` (all versioned 1.00)
- ⬜ Build config loader with startup version-compatibility check
- ⬜ Add HuggingFace token to local `.env` *(DoD: token loads from env, never in code)*
- ⬜ Free disk space if needed; create dedicated `layer_shards_saving_path` folder *(DoD: ≥30 GB free)*
- ⬜ Pull baseline Ollama model (`qwen2.5:7b`) *(DoD: `ollama run` responds)*
- ⬜ Download smoke-test model (Qwen2.5-0.5B) *(DoD: present in HF cache)*

## Phase 2 — Pipeline Smoke-Test  ⬜
*(validate before scaling — the "start small" rule)*
- ⬜ Build SDK skeleton: `run_scenario`, `measure`, `save_result` (+ tests)
- ⬜ Build gatekeeper (rate limit, FIFO queue, retry, logging) reading `rate_limits.json` (+ tests)
- ⬜ Build measurement module: TTFT, ITL/TPOT, throughput, RAM/VRAM samplers (+ tests)
- ⬜ Build recorder: write versioned result JSON to `results/` (+ tests)
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

## Phase 5 — Testing & Quality Gate  ⬜
*(V3 compliance — run before writing the report)*
- ⬜ Backfill any missing tests so every module + public function is covered (happy + error)
- ⬜ `uv run pytest --cov` → **≥85%** global coverage *(DoD: suite green, threshold met)*
- ⬜ `uv run ruff check` → **0** violations *(DoD: clean)*
- ⬜ Confirm no secrets in tree, `.env-example` present, `uv.lock` committed
- ⬜ Confirm every file ≤150 LOC; split any that exceed
- ⬜ Generate automated test report (pass/fail summary) into `reports/`

## Phase 6 — Report (README) & Submission  ⬜
*(est. 1–1.5 h)*
- ⬜ Write deep-dive technical report as `README.md` (hardware, method, baseline, AirLLM, quant, metrics, economics, concept analysis, extension, run instructions)
- ⬜ Embed ALL graphs/tables/screenshots inline in README
- ⬜ Add reproducible run instructions
- ⬜ Document the prompt book (significant prompts used) per V3 §8.3
- ⬜ Add LICENSE + third-party attribution/credits
- ⬜ Push to GitHub (author = user only, no co-author trailer)

---

### Definition of Done (project)
All 8 acceptance criteria in [PRD.md](PRD.md) §3 met; all [PLAN.md](PLAN.md) §2 standards
satisfied (≤150 LOC, SDK, gatekeeper, versioning, ≥85% coverage, 0 Ruff, secrets-clean);
README is a self-contained, reproducible technical report with every required analysis and
all figures embedded.
