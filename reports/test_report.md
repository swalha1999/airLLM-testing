# Automated Test Report

- **Result:** PASS ✅ (pytest exit code 0)
- **Generated:** 2026-06-25T12:36:05.497489+00:00
- **Command:** `uv run python scripts/test_report.py`

The coverage gate (≥85%, config-driven) and the full suite output:

```
..............................................                           [100%]
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.12.13-final-0 _______________

Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src\airllm_bench\__init__.py                2      0   100%
src\airllm_bench\analysis\__init__.py       3      0   100%
src\airllm_bench\analysis\charts.py        37      0   100%
src\airllm_bench\analysis\loader.py        13      0   100%
src\airllm_bench\analysis\metrics.py       12      0   100%
src\airllm_bench\measure\__init__.py        3      0   100%
src\airllm_bench\measure\measurer.py       37      0   100%
src\airllm_bench\measure\resources.py      33      0   100%
src\airllm_bench\sdk\__init__.py            2      0   100%
src\airllm_bench\sdk\sdk.py                28      0   100%
src\airllm_bench\services\__init__.py       2      0   100%
src\airllm_bench\services\costs.py          9      0   100%
src\airllm_bench\shared\__init__.py         5      0   100%
src\airllm_bench\shared\config.py          34      0   100%
src\airllm_bench\shared\gatekeeper.py      74      0   100%
src\airllm_bench\shared\recorder.py        24      0   100%
src\airllm_bench\shared\version.py          3      0   100%
---------------------------------------------------------------------
TOTAL                                     321      0   100%
Required test coverage of 85.0% reached. Total coverage: 100.00%
46 passed in 3.45s
```
