# PLAN — Architecture & Technical Design

**Version:** 1.00
**Companion to:** [PRD.md](PRD.md), [TODO.md](TODO.md)

---

## 1. Approach

A small, focused **experiment harness**: a thin SDK layer exposes the operations the
experiments need (load a model, run a generation, measure it, record results), and the
experiment scripts and analysis notebooks call only that SDK. This keeps measurement logic
in one place so every scenario (baseline / AirLLM / quant levels) is measured *identically*
and is comparable.

The system is deliberately simple — it is a research harness, not a production service — but
it follows the submission guidelines: SDK entry point, no business logic in scripts,
config-driven values, files ≤150 LOC, raw results persisted for reproducible graphs.

## 2. C4 — Context & Containers (text form)

```
        +------------------------------------------------------+
        |                 Experiment Harness                   |
        |                                                      |
  user  |  experiments/      ->   src/sdk (single entry point) |
 ─────► |  (baseline,             |                            |
        |   airllm, quant)        v                            |
        |                    src/runners  ──► HuggingFace Hub  |  (download weights)
        |                    src/measure  ──► Ollama (baseline)|
        |                         |                            |
        |                         v                            |
        |   results/*.json  ◄─ recorder                        |
        |        |                                             |
        |        v                                             |
        |   analysis/  ──►  figures/*.png  ──►  README report  |
        +------------------------------------------------------+
```

- **External systems:** HuggingFace Hub (weights + token), Ollama (local baseline server).
- **Persistence:** raw metrics → `results/` (JSON/CSV); charts → `figures/`.

## 3. Components

| Component | Path (planned) | Responsibility |
|---|---|---|
| SDK | `src/<pkg>/sdk/` | Single entry point: `run_scenario`, `measure`, `save_result` |
| Runners | `src/<pkg>/runners/` | Baseline (Ollama/HF) runner; AirLLM runner; quant config |
| Measurement | `src/<pkg>/measure/` | TTFT, ITL/TPOT, throughput timers; RAM/VRAM samplers (psutil/pynvml) |
| Recorder | `src/<pkg>/shared/` | Serialize results to `results/`, with run metadata + versions |
| Config | `config/` | `setup.json`, model/quant params, cost assumptions (versioned) |
| Constants | `src/<pkg>/constants.py` | Fixed prompts, token counts, paths |
| Analysis | `analysis/` or `notebooks/` | Load `results/`, build figures + break-even |

## 4. Measurement Methodology

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

## 5. Data Schema (results record)

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

## 6. Architecture Decision Records (ADRs)

- **ADR-1 — Package manager: `uv`.** Mandated by guidelines; reproducible via `uv.lock`.
  Alternatives (pip/venv) rejected — explicitly forbidden.
- **ADR-2 — Python 3.11.** torch/transformers/bitsandbytes have no wheels for 3.14 (the
  installed default). 3.11 is broadly supported. Trade-off: not newest, but stable.
- **ADR-3 — Model family: Qwen2.5.** Assignment recommends it; works with `AutoModel`
  (avoids class-mismatch on load). Sizes 0.5B (smoke-test) → 7B (main).
- **ADR-4 — Model size ≤7B.** 44 GB free disk must hold the FP16 download (~15 GB) **plus**
  AirLLM's per-layer sharded copy (~15 GB). 14B+ would overflow disk. Trade-off: smaller
  model, but the RAM bottleneck is still cleanly demonstrated (15 GB model vs 15.7 GB RAM).
- **ADR-5 — Baseline via Ollama.** Already installed; gives a clean, well-optimized direct
  run to contrast against AirLLM. HF direct load used as secondary baseline if useful.
- **ADR-6 — Quantization via AirLLM/bitsandbytes.** Q4/Q8 through AirLLM's compression. Risk:
  bitsandbytes is fragile on Windows/CPU; if it fails, document the failure as a result and
  fall back to whatever quant levels do run.
- **ADR-7 — AirLLM shard path pinned.** Set `layer_shards_saving_path` to a dedicated folder
  to control disk and avoid flooding the OS drive (the "Do" guidance).

## 7. Repository Structure (target)

```
airLLM-testing/
├── README.md                 # the deep-dive technical report (deliverable)
├── pyproject.toml            # uv project + deps
├── uv.lock                   # committed lockfile
├── .gitignore                # excludes .env, weights, shards, caches
├── .env-example              # HF_TOKEN placeholder
├── config/
│   ├── setup.json            # paths, prompts, run params (versioned)
│   └── costs.json            # API prices, wattage, hardware cost (versioned)
├── src/<pkg>/
│   ├── sdk/                  # single entry point
│   ├── runners/              # baseline + airllm runners
│   ├── measure/              # timers + resource samplers
│   ├── shared/               # recorder, config loader, version.py
│   └── constants.py
├── experiments/              # thin run scripts (call the SDK)
├── results/                  # raw metrics (JSON/CSV)
├── figures/                  # generated charts
├── reports/                  # working notes / report drafts
└── docs/                     # PRD, PLAN, TODO, source PDFs
```

> The HW5 PDF allows a lighter structure (`src/ experiments/ results/ reports/ figures/`);
> this plan keeps it light but layers in the SDK + config + security essentials from the
> general guidelines. Structure may be adjusted as long as it stays consistent and navigable.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| bitsandbytes fails on Windows/CPU | Document as result; compare the quant levels that do work |
| Disk overflow from shards | Pin shard path; cap model ≤7B; clear space before download |
| Runs take hours | Start tiny (0.5B, Q2) to validate pipeline before scaling; cap `max_new_tokens` |
| Python 3.14 incompatibility | Pin 3.11 via `uv python install 3.11` |
| Inconsistent measurements | All scenarios go through the same SDK `measure` path |
