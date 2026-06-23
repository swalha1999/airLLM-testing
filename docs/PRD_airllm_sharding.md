# Dedicated PRD — AirLLM Layer-Sharding Mechanism

**Version:** 1.00
**Parent:** [PRD.md](PRD.md) · **Design:** [PLAN.md](PLAN.md)
**Mechanism:** the core technique that lets an oversized model run on a memory-constrained machine.

---

## 1. Description & Theoretical Background

A transformer LLM is a stack of N near-identical decoder layers. In a normal forward pass,
**all** layer weights must be resident in memory simultaneously — so peak memory ≈ full model
size. On a 15.7 GB-RAM laptop, a 15 GB FP16 model leaves no headroom and triggers swapping
or an out-of-memory failure.

**AirLLM** breaks this constraint with **layer-by-layer execution**:

1. The model is split (sharded) on disk into one weight file per transformer layer
   (`layer_shards_saving_path`), stored as SafeTensors.
2. At inference, AirLLM loads **only the current layer** into RAM, runs the activation
   through it, then frees it and loads the next layer.
3. Peak memory therefore scales with the **largest single layer**, not the whole model.

This is a direct analogue of **virtual memory / demand paging**: the "working set" held in
fast memory at any instant is one layer, while the rest of the model lives on the slower disk
"backing store" and is paged in on demand. The cost is **I/O**: every token's forward pass
re-reads the entire model from disk, layer by layer.

## 2. Requirements

| ID | Requirement |
|---|---|
| R1 | Load a model whose full size exceeds available RAM, via AirLLM `AutoModel`. |
| R2 | Pin `layer_shards_saving_path` to a dedicated fast-drive folder (not the OS drive root). |
| R3 | Run the same fixed prompt/`max_new_tokens` used by every other scenario. |
| R4 | Capture peak RAM during the run to confirm it tracks one layer, not the full model. |
| R5 | Capture the disk-I/O / wall-clock cost that the paging analogy predicts (slow TTFT/TPOT). |
| R6 | Work with the Qwen2.5 family through the generic `AutoModel` to avoid class-mismatch. |

## 3. Expected I/O

- **Input:** model id (e.g. `Qwen/Qwen2.5-7B`), compression level, prompt, `max_new_tokens`, shard path.
- **Output:** generated text **plus** a results record (TTFT, TPOT, throughput, peak RAM/VRAM,
  runtime) written to `results/`.
- **Side effect:** per-layer SafeTensors written once to the shard path (reused on later runs).

## 4. Performance Metrics

- Peak RAM (expect **far below** full model size — the headline evidence).
- TTFT and TPOT (expect **high** — dominated by per-layer disk reads → memory-bound).
- Throughput in tok/s (expect **low**; quantify the latency "price").
- First-run vs warm-run delta (sharding write cost amortized after run 1).

## 5. Constraints & Limitations

- Disk must hold the original download **and** the sharded copy (~2× model size). Caps us at ≤7B given ~44 GB free.
- Each generated token re-reads the whole model from disk → inherently slow; this is expected, not a bug.
- A 2 GB GPU cannot hold a layer meaningfully → execution is effectively CPU + disk bound.

## 6. Alternatives Considered & Rationale

| Alternative | Why not (here) |
|---|---|
| Load full model in RAM | Doesn't fit / swaps unusably — this is the baseline failure we contrast against. |
| `accelerate` CPU/disk offload | Similar idea; AirLLM is the assignment's named tool and gives explicit per-layer control. |
| GPU-only quantized load | 2 GB VRAM is far too small for a 7B model. |

**Chosen:** AirLLM layer-sharding — it is the assignment's subject and the only approach that
makes an oversized model runnable on this hardware while exposing the paging behaviour we analyze.

## 7. Success Criteria & Test Scenarios

- **SC1:** model that fails/swaps in the baseline completes a generation under AirLLM. *(negative result acceptable if analyzed)*
- **SC2:** measured peak RAM is well under full model size (paging confirmed).
- **SC3:** TTFT/TPOT clearly higher than baseline where the baseline ran at all — quantifying the trade.
- **Test scenarios:** (a) smoke-test on Qwen2.5-0.5B confirms sharding+load path; (b) main run on Qwen2.5-7B; (c) shard-path misconfiguration handled with a clear error (defensive check).
