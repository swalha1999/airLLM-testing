"""Original extension — break-even sensitivity analysis (OAT) (#26, FR-g).

How robust is the ~7,691 req/month break-even (#23) to its assumptions? This does
a one-at-a-time (OAT) sweep: vary each key input (API output price, electricity
tariff, hardware CAPEX) from 0.5x to 2x baseline, recompute the break-even with
the tested cost model, and plot all three curves on one axis. The steeper a curve,
the more sensitive the decision is to that assumption (V3 §9.1).

Run with: uv run python experiments/sensitivity.py
"""

from airllm_bench.analysis.charts import line_chart
from airllm_bench.services.costs import (
    api_cost_per_request,
    break_even_requests,
    electricity_per_request,
    onprem_fixed_monthly,
)
from airllm_bench.shared.config import ConfigLoader
from airllm_bench.shared.recorder import ResultRecorder

MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
FIGURE = "figures/sensitivity.png"


def main() -> None:
    """Sweep each assumption OAT, chart the break-even curves, record them."""
    cfg = ConfigLoader().load("costs", "1.10").data
    work, onprem, api = cfg["workload"], cfg["on_prem"], cfg["api_providers"]["gpt-4o"]
    in_tok, out_tok = work["input_tokens_per_request"], work["output_tokens_per_request"]
    runtime_s = out_tok / work["local_throughput_tok_s"]

    def break_even(*, out_price: float, kwh: float, capex: float) -> float | None:
        api_req = api_cost_per_request(in_tok, out_tok, api["input_usd_per_1m_tokens"], out_price)
        fixed = onprem_fixed_monthly(
            capex, onprem["hardware_lifetime_months"], onprem["maintenance_usd_per_month"]
        )
        elec = electricity_per_request(runtime_s, onprem["device_power_watts"], kwh)
        return break_even_requests(fixed, api_req, elec)

    base = {
        "out_price": api["output_usd_per_1m_tokens"],
        "kwh": onprem["electricity_usd_per_kwh"],
        "capex": onprem["hardware_capex_usd"],
    }
    curves = {
        "API output price": [
            break_even(**{**base, "out_price": base["out_price"] * m}) for m in MULTIPLIERS
        ],
        "Electricity tariff": [break_even(**{**base, "kwh": base["kwh"] * m}) for m in MULTIPLIERS],
        "Hardware CAPEX": [break_even(**{**base, "capex": base["capex"] * m}) for m in MULTIPLIERS],
    }

    line_chart(
        {label: (MULTIPLIERS, values) for label, values in curves.items()},
        "Break-even sensitivity (OAT) — one assumption varied at a time (cross at 1x baseline)",
        "assumption multiplier (x baseline)",
        "break-even (requests / month)",
        FIGURE,
    )

    record = {"scenario": "break_even_sensitivity", "multipliers": MULTIPLIERS, "curves": curves}
    for label, values in curves.items():
        print(f"{label}: {[round(v) if v else None for v in values]}")
    print(f"\nrecorded: {ResultRecorder().save(record)}\nfigure: {FIGURE}")


if __name__ == "__main__":
    main()
