"""FR-e — On-Prem vs API break-even analysis (#23).

Computes the monthly request volume at which running locally becomes cheaper than
a hosted API, from the assumptions in ``config/costs.json`` (no hard-coded numbers,
V3 S3). Uses the *practical* local engine throughput (Ollama Q4) for the
per-request electricity cost — AirLLM FP16 at ~0.01 tok/s would be hours/request
and is noted separately as impractical. Writes a cost-vs-volume chart + a record.

Run with: uv run python experiments/break_even.py
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

PROVIDER = "gpt-4o"
FIGURE = "figures/break_even.png"


def main() -> None:
    """Compute the break-even, chart cost-vs-volume, and record the result."""
    cfg = ConfigLoader().load("costs", "1.10").data
    work, onprem, api = cfg["workload"], cfg["on_prem"], cfg["api_providers"][PROVIDER]

    in_tok = work["input_tokens_per_request"]
    out_tok = work["output_tokens_per_request"]
    throughput = work["local_throughput_tok_s"]

    api_per_req = api_cost_per_request(
        in_tok, out_tok, api["input_usd_per_1m_tokens"], api["output_usd_per_1m_tokens"]
    )
    fixed = onprem_fixed_monthly(
        onprem["hardware_capex_usd"],
        onprem["hardware_lifetime_months"],
        onprem["maintenance_usd_per_month"],
    )
    elec_per_req = electricity_per_request(
        out_tok / throughput, onprem["device_power_watts"], onprem["electricity_usd_per_kwh"]
    )
    break_even = break_even_requests(fixed, api_per_req, elec_per_req)

    top = int((break_even or 1000) * 2)
    step = max(top // 50, 1)
    volumes = list(range(0, top + 1, step))
    line_chart(
        {
            "API (gpt-4o)": (volumes, [v * api_per_req for v in volumes]),
            "On-Prem (local)": (volumes, [fixed + v * elec_per_req for v in volumes]),
        },
        "Monthly cost vs request volume (On-Prem vs API)",
        "requests / month",
        "USD / month",
        FIGURE,
        vline=break_even,
    )

    record = {
        "scenario": "break_even",
        "provider": PROVIDER,
        "assumptions": {
            "input_tokens_per_request": in_tok,
            "output_tokens_per_request": out_tok,
            "local_throughput_tok_s": throughput,
            **onprem,
        },
        "api_cost_per_request_usd": round(api_per_req, 6),
        "onprem_fixed_monthly_usd": round(fixed, 2),
        "onprem_electricity_per_request_usd": round(elec_per_req, 6),
        "break_even_requests_per_month": round(break_even) if break_even else None,
    }
    print(f"API ${api_per_req:.5f}/req | On-Prem ${fixed:.2f}/mo + ${elec_per_req:.5f}/req")
    print(f"break-even ~ {record['break_even_requests_per_month']} requests/month")
    print(f"\nrecorded: {ResultRecorder().save(record)}\nfigure: {FIGURE}")


if __name__ == "__main__":
    main()
