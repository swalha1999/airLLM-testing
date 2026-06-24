"""On-Prem vs API cost model and break-even (FR-e, PRD_cost_model).

Two cost shapes: an API is pure **variable** cost (pay per token, every request),
while on-prem is mostly **fixed** (amortized hardware + maintenance) plus a small
per-request electricity cost. They cross at the break-even request volume — below
it the API is cheaper, above it on-prem wins. All inputs come from
``config/costs.json`` (V3 S3); these are just the formulas.
"""


def api_cost_per_request(
    input_tokens: float,
    output_tokens: float,
    input_usd_per_1m: float,
    output_usd_per_1m: float,
) -> float:
    """Cost of one API request given token counts and per-million prices."""
    return input_tokens / 1e6 * input_usd_per_1m + output_tokens / 1e6 * output_usd_per_1m


def onprem_fixed_monthly(
    capex_usd: float,
    lifetime_months: float,
    maintenance_usd_per_month: float,
) -> float:
    """Fixed monthly on-prem cost: amortized hardware + maintenance."""
    return capex_usd / lifetime_months + maintenance_usd_per_month


def electricity_per_request(runtime_s: float, watts: float, usd_per_kwh: float) -> float:
    """Electricity cost of one local request (runtime x power x tariff)."""
    return runtime_s / 3600.0 * watts / 1000.0 * usd_per_kwh


def break_even_requests(
    fixed_monthly: float,
    api_per_request: float,
    onprem_var_per_request: float,
) -> float | None:
    """Monthly request volume where on-prem total equals API total.

    Returns ``None`` when on-prem's per-request cost is not below the API's (the
    lines never cross, so on-prem never pays back its fixed cost).
    """
    margin = api_per_request - onprem_var_per_request
    return fixed_monthly / margin if margin > 0 else None
