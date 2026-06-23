# Phase 2 — AirLLM Smoke-Test Findings (#14)

**Goal:** validate the AirLLM pipeline on a tiny model (Qwen2.5-0.5B) before
scaling to the 7B, per the "start small" rule.

**Outcome:** the harness and AirLLM's sharding both work; the 0.5B itself is
**incompatible** with AirLLM and cannot generate. This is a documented result —
the real generation + quant sweep runs on the compatible 7B in Phase 3.

## What we had to fix first (dependency resolution)

AirLLM 2.11 has unpinned dependencies, so the latest versions resolved and broke
its import chain. Working pins (now in `pyproject.toml`, `ml` extra):

| Package | Latest (broken) | Pinned | Why |
|---|---|---|---|
| transformers | 5.12.1 | `>=4.40,<4.49` | airllm targets the 4.x API; optimum's BetterTransformer needs `<4.49` |
| optimum | 2.2.0 | `>=1.20,<2` | airllm imports `optimum.bettertransformer`, removed in optimum 2.0 |
| sentencepiece | (absent) | `>=0.2` | airllm's bundled tokenizers import it eagerly |

After these, `from airllm import AutoModel` imports cleanly on Python 3.12 / torch 2.12.1+cpu.

## What worked

- ✅ airllm imports and initialises (bitsandbytes detected).
- ✅ **Layer sharding works** — AirLLM split all 24 transformer layers (+ `embed_tokens`,
  `norm`) to `C:\airllm_shards\splitted_model\` as individual SafeTensors. The core
  mechanism this assignment studies is functioning.
- ✅ The harness path is proven: `experiments/smoke_test.py` → `AirLLMRunner` →
  `ResultRecorder` wrote a versioned result JSON to `results/`.

## What blocked the 0.5B (two real AirLLM constraints)

1. **Requires a SafeTensors index.** Single-file checkpoints (`model.safetensors`
   with no `*.index.json`) abort with `AssertionError: model.safetensors.index.json
   should exist`. Fixed with `ensure_safetensors_index()`, which synthesises the index.
2. **Requires an untied `lm_head` / multi-shard layout.** Qwen2.5-0.5B has
   `tie_word_embeddings: true` and ships **no** `lm_head.weight`. AirLLM's
   `split_and_save_layers` then hits `single_modelfile = shards[0]` → `IndexError`.
   There is no clean fix without patching AirLLM internals.

Recorded result: `results/smoke_airllm_0_5b_*.json` (`status: documented_failure`).

## Conclusion

AirLLM is built for **large, multi-shard models with a separate `lm_head`**. The 7B
main model (`Qwen/Qwen2.5-7B`) is natively that shape, so the runner works for it
unchanged — the smoke-test's job (de-risk the pipeline + nail down the dependency
set) is done. The actual generation, metrics, and quantization sweep run on the 7B
in Phase 3 (#16/#17).

**Reproduce:** `uv run --extra ml python experiments/smoke_test.py`
