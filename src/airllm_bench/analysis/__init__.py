"""Analysis layer — load saved results and build comparisons/figures."""

from .loader import latest_by_scenario, load_results
from .metrics import METRIC_FIELDS, metrics_row, metrics_table

__all__ = [
    "METRIC_FIELDS",
    "latest_by_scenario",
    "load_results",
    "metrics_row",
    "metrics_table",
]
