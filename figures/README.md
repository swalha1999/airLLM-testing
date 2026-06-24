# figures/ — Comparison Charts (#22)

Generated from the saved result records by `experiments/make_figures.py` (no re-running).
Regenerate with: `uv run python experiments/make_figures.py`.

| Figure | Shows |
|---|---|
| `throughput_comparison.png` | Throughput (tok/s, log) — Ollama Q4 (0.79) vs AirLLM FP16 (0.010); Q4 ~79× faster |
| `peak_ram_comparison.png` | Peak process RAM, same FP16 7B — HF direct (11.7 GB + ~15 GB swap) vs AirLLM (7.74 GB, no swap) |
| `runtime_comparison.png` | Total runtime (s, log) — Ollama 55 s, AirLLM 798 s, HF direct 8,070 s (differing token counts) |

Charts use a consistent palette and value labels (V3 §9.3) and are embedded in the README report.
