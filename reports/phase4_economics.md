# Phase 4 — Economics: On-Prem vs API Break-Even (FR-e, #23)

**Goal:** decide when running locally is cheaper than a hosted API, as a function of usage.

**Headline:** with our assumptions, **on-prem breaks even at ~7,700 requests/month**. Below
that the API is cheaper; above it, local wins. But this only holds for a *fast* local engine —
AirLLM FP16 at ~0.01 tok/s is hours per request and never makes economic sense here.

![Cost vs request volume, break-even ~7,691/month](../figures/break_even.png)

## The two cost shapes

- **API** — pure variable cost: a steep line from the origin (pay per token, every request).
- **On-Prem** — mostly fixed: amortized hardware + maintenance, near-flat, with a tiny
  per-request electricity cost.

They cross at the break-even volume.

## The numbers (from `config/costs.json`, via `experiments/break_even.py`)

| Quantity | Value | Source / formula |
|---|---|---|
| Workload per request | 500 in + 500 out tokens | assumption |
| **API** (gpt-4o) | **$0.00625 / request** | 500·$2.5/M + 500·$10/M |
| On-prem fixed | **$38.33 / month** | $1200 / 36 mo + $5 maintenance |
| On-prem electricity | **$0.00127 / request** | (500 / 0.79 tok/s) · 45 W · $0.16/kWh |
| **Break-even** | **≈ 7,691 requests / month** | $38.33 / ($0.00625 − $0.00127) |

Recorded: `results/break_even_*.json`.

## Every assumption stated (so it's reproducible)

- API price: gpt-4o snapshot ($2.50 / $10.00 per 1M in/out tokens).
- Hardware: $1,200 amortized over 36 months; $5/month maintenance.
- Power: 45 W device, $0.16/kWh; runtime = output_tokens / local throughput.
- **Local throughput = 0.79 tok/s** — the *practical* engine (Ollama Q4). All values live
  in `config/costs.json`; change them there and the break-even recomputes.

## The crucial caveat — which local engine?

The electricity-per-request depends entirely on **how fast** the local model runs:

- **Ollama Q4 (0.79 tok/s):** 500 tokens ≈ 10.5 min/request → $0.0013/request. Break-even
  ~7,700/month — **realistic**.
- **AirLLM FP16 (0.01 tok/s):** 500 tokens ≈ **14 hours/request** → impractical at any volume.

So "on-prem is cheaper above ~7,700 req/month" is true **only with a fast quantized local
engine**. The feasibility tool that lets the big model run at all (AirLLM) is *not* the tool
that makes it economic.

## Recommendation

- **Low / bursty volume (< ~7,700 req/month) or strict latency needs → API.** No upfront
  cost, and far faster per request.
- **High, steady volume (> ~7,700 req/month) with a fast local engine → on-prem**, especially
  when **data privacy / control** matter (a non-price factor that can dominate the decision —
  on-prem keeps prompts and data in-house). Prompt/context caching on the API side would push
  the break-even higher for repetitive workloads.

**Reproduce:** `uv run python experiments/break_even.py`
