# PLAN — Architecture & Technical Design

**Version:** 1.00
**Companion to:** [PRD.md](PRD.md), [TODO.md](TODO.md)
**Compliance target:** Full adherence to *Submission Guidelines V3* (see §2).

---

## 1. Approach

A small, focused **experiment harness**: a thin SDK layer exposes the operations the
experiments need (load a model, run a generation, measure it, record results), and the
experiment scripts and analysis notebooks call only that SDK. This keeps measurement logic
in one place so every scenario (baseline / AirLLM / quant levels) is measured *identically*
and is comparable.

The system is deliberately simple — a research harness, not a production service — but it is
built to satisfy **every enforced V3 standard** in §2 below.

## 2. Coding Standards & Constraints (Enforced)

These are hard rules for all code in this project, applied at **full V3 compliance**. Each is
listed once here so it is explicit rather than scattered.

| # | Standard | Rule | Enforcement |
|---|---|---|---|
| S1 | **File size** | Every code file ≤ **150 lines** (blanks/comments excluded). Over → split, never compress. | manual + review |
| S2 | **Package manager** | `uv` only. No `pip` / `venv` / `python -m`. `uv.lock` committed. | `uv` in all docs/scripts |
| S3 | **No hardcoded values** | Configurable values come from `config/*.json`; secrets from env. Allowed in code: math constants, defaults, `constants.py`, Enums. | review |
| S4 | **Secrets** | No keys/tokens in code. `.env` git-ignored; `.env-example` committed. `.gitignore` covers `.env`, `*.key`, `*.pem`. | scan |
| S5 | **SDK architecture** | All business logic behind the SDK layer. Scripts/notebooks/CLI delegate to SDK; no logic in them. | review |
| S6 | **OOP / DRY** | No duplication. Same logic in 2+ files → shared module/base class/mixin. Each mixin = one concern, independently testable. | review |
| S7 | **API gatekeeper** | Every external API call (cost-analysis provider calls, HF Hub if scripted) goes through a centralized gatekeeper: rate-limit check, FIFO overflow queue, retries, logging. Limits from `config/rate_limits.json`. | review + test |
| S8 | **Versioning** | `src/<pkg>/shared/version.py` `__version__ = "1.00"`; every JSON config has a `"version"` key starting `1.00`. Validate config version at startup. | startup check |
| S9 | **TDD & coverage** | Red→Green→Refactor. Every module has a test file; every public function ≥1 test (happy + error path). External deps mocked. **Global coverage ≥ 85%**, suite fails below. Test files also ≤150 lines. | `uv run pytest --cov` |
| S10 | **Linting** | **Zero** Ruff violations. Rule set `E,F,W,I,N,UP,B,C4,SIM`. | `uv run ruff check` |
| S11 | **Docs/comments** | Docstrings on every module/class/function; comments explain *why*. Descriptive names; short single-responsibility functions. | review |
| S12 | **Package layout** | `__init__.py` in package + every subdir, exposing public API via `__all__`. Imports relative / by package name — never absolute. File I/O relative to package. | review |
| S13 | **Building blocks** | Each component documents Input / Output / Setup; validates inputs; testable via dependency injection. | review |
| S14 | **Parallelism safety** | Resource samplers run in threads → protect shared state with locks, use `queue.Queue`, context managers, no leaks. | review |

> **Note on fit:** S7 (gatekeeper) and S9 (85% coverage) are written for production apps. We
> apply them fully anyway: the gatekeeper wraps the cost-analysis API calls and any scripted
> HF downloads; coverage targets the harness logic (SDK, measurement, recorder, cost model,
> gatekeeper) with heavy model I/O mocked so tests never depend on external services.

## 3. C4 — Context & Containers (text form)

```
        +------------------------------------------------------+
        |                 Experiment Harness                   |
        |                                                      |
  user  |  experiments/      ->   src/sdk (single entry point) |
 ─────► |  (baseline,             |                            |
        |   airllm, quant)        v                            |
        |                    src/runners  ──┐                  |
        |                    src/measure    ├─► gatekeeper ──► HuggingFace Hub
        |                         |         └──────────────► Ollama / cost API
        |                         v                            |
        |   results/*.json  ◄─ recorder                        |
        |        |                                             |
        |        v                                             |
        |   analysis/  ──►  figures/*.png  ──►  README report  |
        +------------------------------------------------------+
```

- **External systems:** HuggingFace Hub (weights + token), Ollama (local baseline), cost-API provider (pricing reference).
- **All external calls pass through the gatekeeper (S7).**
- **Persistence:** raw metrics → `results/` (JSON/CSV); charts → `figures/`.

## 4. Components

| Component | Path (planned) | Responsibility |
|---|---|---|
| SDK | `src/<pkg>/sdk/` | Single entry point: `run_scenario`, `measure`, `save_result` |
| Runners | `src/<pkg>/runners/` | Baseline (Ollama/HF) runner; AirLLM runner; quant config |
| Measurement | `src/<pkg>/measure/` | TTFT, ITL/TPOT, throughput timers; RAM/VRAM samplers (psutil/pynvml) |
| Gatekeeper | `src/<pkg>/shared/gatekeeper.py` | Centralized external-call manager: rate limit, queue, retry, log |
| Recorder | `src/<pkg>/shared/recorder.py` | Serialize results to `results/` with run metadata + versions |
| Config | `src/<pkg>/shared/config.py` + `config/` | Load & version-check `setup.json`, `costs.json`, `rate_limits.json` |
| Version | `src/<pkg>/shared/version.py` | `__version__ = "1.00"` |
| Constants | `src/<pkg>/constants.py` | Fixed prompts, token counts, paths |
| Cost model | `src/<pkg>/services/costs.py` | Break-even / CAPEX+OPEX vs API math |
| Analysis | `notebooks/` / `analysis/` | Load `results/`, build figures + break-even |
| Tests | `tests/unit`, `tests/integration` | Mirror `src/`; mocks for model/API; `conftest.py` fixtures |

## 5. Measurement Methodology

- **TTFT (Time To First Token):** wall-clock from request to first streamed token. Proxy
  for Prefill cost + KV-cache build.
- **ITL / TPOT:** mean inter-token latency after the first token (ms/token). Proxy for
  Decode-step memory traffic.
- **Throughput:** total output tokens / total generation time (tok/s).
- **Peak RAM:** sampled via `psutil` in a background thread during the run.
- **Peak VRAM:** sampled via `pynvml` (best-effort; near-zero given 2 GB GPU).
- **Total runtime & estimated power:** wall-clock × assumed wattage (stated in config).
- **Output quality:** fixed prompt set, qualitative rubric per quant level.
- Controls: identical prompts, identical `max_new_tokens`, warm vs cold noted, raw numbers
  always saved for re-plotting.

## 6. Data Schema (results record)

```json
{
  "schema_version": "1.00",
  "scenario": "airllm_q4",
  "model": "Qwen/Qwen2.5-7B",
  "quant": "Q4",
  "prompt_id": "long_context_01",
  "metrics": {
    "ttft_s": 12.3,
    "tpot_ms": 850.0,
    "throughput_tok_s": 1.18,
    "peak_ram_gb": 6.4,
    "peak_vram_gb": 0.1,
    "total_runtime_s": 190.0,
    "est_power_wh": 9.5
  },
  "output_tokens": 128,
  "timestamp": "..."
}
```

## 7. Architecture Decision Records (ADRs)

- **ADR-1 — Package manager: `uv`.** Mandated; reproducible via `uv.lock`. pip/venv rejected (forbidden).
- **ADR-2 — Python 3.12.** torch/transformers/bitsandbytes have no wheels for 3.14 (installed default). 3.12 is fully supported by torch 2.2+ and the rest of the stack; pinned via `.python-version`.
- **ADR-3 — Model family: Qwen2.5.** Recommended; works with `AutoModel` (avoids class-mismatch). 0.5B (smoke-test) → 7B (main).
- **ADR-4 — Model size ≤7B.** 44 GB free disk must hold FP16 download (~15 GB) **plus** AirLLM's per-layer shards (~15 GB). 14B+ overflows. 15 GB model vs 15.7 GB RAM still demonstrates the bottleneck cleanly.
- **ADR-5 — Baseline via Ollama.** Installed; clean optimized direct run to contrast with AirLLM. HF direct load as secondary baseline.
- **ADR-6 — Quantization via AirLLM/bitsandbytes.** Q4/Q8 via AirLLM compression. Risk: bitsandbytes fragile on Windows/CPU; failure is documented as a result; fall back to working quant levels.
- **ADR-7 — AirLLM shard path pinned.** `layer_shards_saving_path` → dedicated folder to control disk and avoid flooding the OS drive.
- **ADR-8 — Full V3 compliance.** Per user decision, apply all §2 standards including gatekeeper (S7) and 85% coverage (S9), scoped as noted, even though the project is an experiment not an app.

## 8. Repository Structure (target)

```
airLLM-testing/
├── README.md                 # the deep-dive technical report (deliverable)
├── pyproject.toml            # uv project, deps, ruff + coverage config
├── uv.lock                   # committed lockfile
├── .gitignore                # excludes .env, weights, shards, caches
├── .env-example              # HF_TOKEN placeholder
├── config/
│   ├── setup.json            # paths, prompts, run params (versioned)
│   ├── costs.json            # API prices, wattage, hardware cost (versioned)
│   └── rate_limits.json      # gatekeeper limits (versioned)
├── src/<pkg>/
│   ├── __init__.py           # exposes public API, __version__
│   ├── sdk/                  # single entry point
│   ├── runners/              # baseline + airllm runners
│   ├── measure/              # timers + resource samplers
│   ├── services/             # cost model, orchestration
│   ├── shared/               # gatekeeper, config, recorder, version.py
│   └── constants.py
├── tests/
│   ├── unit/                 # mirrors src/
│   ├── integration/
│   └── conftest.py           # shared fixtures, mocks
├── experiments/              # thin run scripts (call the SDK)
├── results/                  # raw metrics (JSON/CSV)
├── figures/                  # generated charts
├── notebooks/                # analysis notebook
├── reports/                  # working notes / report drafts
└── docs/                     # PRD, PLAN, TODO, dedicated PRDs, source PDFs
```

> The HW5 PDF allows a lighter structure; this plan adopts the **full** V3 package layout
> (SDK, config, tests, gatekeeper) on top of it. May be adjusted while staying consistent.

## 9. Dedicated PRDs (per V3 §2.3)

Central mechanisms get their own PRD under `docs/`:
- `docs/PRD_airllm_sharding.md` — layer-sharding mechanism, theory, I/O, perf metrics.
- `docs/PRD_quantization.md` — quantization (FP16/Q8/Q4), accuracy/memory trade-offs.
- `docs/PRD_cost_model.md` — break-even model: inputs, formulas, assumptions.

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| bitsandbytes fails on Windows/CPU | Document as result; compare quant levels that work |
| Disk overflow from shards | Pin shard path; cap model ≤7B; clear space before download |
| Runs take hours | Start tiny (0.5B, Q2) to validate pipeline before scaling; cap `max_new_tokens` |
| Python 3.14 incompatibility | Pin 3.12 via `uv python install 3.12` + `.python-version` |
| Inconsistent measurements | All scenarios go through the same SDK `measure` path |
| 85% coverage hard on model I/O | Mock model/API in tests; cover harness logic (SDK, measure, recorder, costs, gatekeeper) |
