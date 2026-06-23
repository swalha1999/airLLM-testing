# Dedicated PRD — Economic Cost Model (On-Prem vs API Break-Even)

**Version:** 1.00
**Parent:** [PRD.md](PRD.md) · **Design:** [PLAN.md](PLAN.md)
**Mechanism:** the math that answers "when is running locally cheaper than calling a hosted API?"

---

## 1. Description & Theoretical Background

Two ways to serve an LLM workload have opposite cost shapes:

- **External API** — pure **variable** cost: you pay per token (input + output) every request.
  Zero up-front, but cost grows linearly with usage forever.
- **On-Prem (local)** — mostly **fixed** cost: hardware (**CAPEX**, amortized over its life)
  plus electricity + maintenance (**OPEX**). Near-flat regardless of how many requests you run.

Plotting cumulative cost vs usage volume, the API is a steep line from the origin; on-prem is
a low-slope line starting above zero. They cross at the **break-even point** — the usage volume
beyond which local becomes cheaper. Below it, the API wins; above it, on-prem wins.

A realistic model must also consider **Prompt/Context Caching** (e.g. KV-cache reuse offered by
some API providers via paged attention), which discounts repeated prompt prefixes and can shift
the break-even point on repetitive, long-context workloads.

## 2. Requirements

| ID | Requirement |
|---|---|
| R1 | Compute API cost = (input_tokens + output_tokens) × provider price-per-token, per request and per volume. |
| R2 | Compute on-prem cost = amortized CAPEX + OPEX (electricity from measured runtime × wattage + maintenance). |
| R3 | Derive the **break-even** usage volume where the two are equal. |
| R4 | Plot cumulative cost vs usage volume (both lines; break-even marked). |
| R5 | State **every** assumption explicitly (prices, volume, hardware life, wattage, electricity tariff) so the analysis is transparent and reproducible. |
| R6 | Give a reasoned recommendation: which scenarios favor API, which favor on-prem (include privacy/security/data-control, not just price). |
| R7 | Optional third line: Cloud-GPU rental (per-GPU-hour × runtime). |
| R8 | All numeric inputs live in `config/costs.json` (versioned) — no hardcoding. |

## 3. Expected I/O

- **Input:** measured runtime + tokens from `results/`; assumptions from `config/costs.json`
  (per-token price, hardware cost, hardware lifetime, watts, $/kWh, maintenance).
- **Output:** a cost table (per-request + per-volume), the break-even volume, and a
  cumulative-cost-vs-volume figure in `figures/`.

## 4. Performance / Output Metrics

- Cost per request (API) and effective cost per request (on-prem) as a function of volume.
- Break-even volume (requests, or tokens, per period).
- Sensitivity: how break-even moves if price / wattage / hardware-life assumptions change (OAT).

## 5. Constraints & Limitations

- Provider prices change; the model is a snapshot with the date + source recorded.
- On-prem ignores hard-to-quantify factors (developer time, reliability) — noted as caveats.
- Power is **estimated** (runtime × assumed wattage), not wall-meter measured — stated clearly.

## 6. Alternatives Considered & Rationale

| Alternative | Why not (here) |
|---|---|
| Single point-cost comparison | Misses the whole story; break-even-vs-volume is what the assignment asks for. |
| Ignore caching | Less realistic; prompt caching can materially move break-even — included as a factor. |
| Hardcode prices in code | Violates V3 S3; all assumptions go in `config/costs.json`. |

**Chosen:** a transparent break-even model driven entirely by a versioned config file, plotted
as cumulative cost vs volume with all assumptions printed alongside.

## 7. Success Criteria & Test Scenarios

- **SC1:** break-even volume computed and shown on a labeled cost-vs-volume chart.
- **SC2:** every assumption is listed and sourced; result is reproducible from `config/costs.json`.
- **SC3:** recommendation distinguishes which workloads favor API vs on-prem (incl. privacy).
- **Test scenarios:** (a) unit tests on the cost formulas with known inputs → known break-even;
  (b) zero-usage edge case (API = 0, on-prem = fixed); (c) caching-on vs caching-off shifts break-even as expected.
