# Phase 4 — Concept Analysis (FR-f, #24)

Tying the measured results back to LLM-inference theory — the *why* behind the numbers.

## 1. Prefill vs Decode → TTFT vs TPOT

A generation has two phases:

- **Prefill** — the prompt's tokens are processed in one parallel pass to build the KV-cache.
  Its cost shows up as **TTFT** (time to first token).
- **Decode** — tokens are produced one at a time, each reading the whole model + the growing
  KV-cache. Its cost shows up as **TPOT** (time per output token).

In our clean run (`baseline_ollama_q4`): **TTFT = 46 s**, **TPOT = 206 ms**. The TTFT is huge
relative to TPOT — but almost all of it is the **one-time model load** on the first call, not
Prefill compute (the prompt is tiny). This is itself the lesson: on memory-constrained local
hardware, **getting the model resident dominates first-token latency**, swamping the actual
Prefill. Once loaded, steady-state Decode (206 ms/token) is comparatively cheap.

For the FP16 runs we deliberately measured peak RAM + total runtime rather than streaming
TTFT/TPOT, because there the *memory* story is the point (see §3).

## 2. The role of VRAM (and its absence here)

VRAM is fast memory attached to the GPU; it's where weights and the KV-cache normally live for
fast inference. This machine has a 2 GB MX350 and a CPU-only torch build, so **VRAM is
effectively zero for our purposes** — every run sat on **system RAM + CPU**, GPU idle (0% in
the Task Manager shots). That single hardware fact shapes everything below: with no VRAM to
hold the model, the only question is whether it fits in **system RAM**, and if not, what gives.

## 3. Memory-bound vs compute-bound

A workload is **compute-bound** if the processor is the limit, **memory-bound** if moving data
is. Every result here is firmly **memory-bound**:

- During the FP16 baseline, **CPU sat ~33–100% but Disk was pegged at 100%** and RAM at 99%
  with 238 MB free. The CPU was *waiting on data*, not computing — the definition of
  memory-bound. The 7B in FP16 needs ~15 GB of weights moved per forward pass; with 15.7 GB
  total RAM there's no room, so the bottleneck is memory **capacity** then memory/disk
  **bandwidth**.
- The quantization result confirms it: ~Q4 moves ¼ the bytes and runs ~79× faster (#18). If
  the work were compute-bound, fewer bits wouldn't help nearly that much — the win comes from
  **moving less data**, the signature of a memory-bound regime.
- AirLLM's CPU also idled (~33%) while it streamed layers from disk — again, **I/O-bound**, a
  flavour of memory-bound where the "memory" is the SSD.

A roofline view (#25) places all scenarios on the memory-bandwidth slope, never near the
compute ceiling.

## 4. The virtual-memory / paging analogy (the heart of it)

This is the key concept the project demonstrates twice — once badly, once well:

- **Baseline FP16 direct = uncontrolled OS swap.** Loading 15 GB into 15.7 GB RAM forces the
  OS to **page** weights between RAM and the SSD on demand. Committed memory hit **30.6 GB**
  against 15.7 GB physical — roughly half the model living in the page file. The OS thrashes
  blindly (it doesn't know the access pattern), so the run crawls for **2 h 14 min**. This is
  virtual memory failing: the working set is larger than RAM and the pager flails.
- **AirLLM = manual, ordered paging of layers.** AirLLM does the *same* thing — keep most of
  the model on disk, bring pieces into RAM as needed — but **deliberately and in order**: load
  layer *i*, compute, free it, load layer *i+1* (prefetching the next). The working set is one
  (prefetched: ~two) layer, so **peak RAM stays at 7.74 GB with zero swap** (Task Manager: 50%
  RAM, 21% disk, 8 GB free). Same model, same precision, same disk-bound nature — but a
  *controlled* paging schedule instead of chaotic thrashing.

So AirLLM is essentially a **purpose-built pager for transformer layers**: it accepts the
memory-bound reality and manages it, trading predictable disk I/O per token for a tiny, stable
memory footprint. It does not make the model *fast* — it makes it *fit*. That distinction —
feasibility, not performance — is the whole point, and the data shows it: AirLLM is ~79× slower
than the in-memory Q4 engine, yet it is the only way the FP16 model runs at all on this
machine.

## 5. One-line synthesis

The bottleneck was **memory, not compute**; quantization helps because it **moves fewer
bytes**; and AirLLM helps because it **turns ruinous OS swap into orderly layer paging** —
buying feasibility on hardware that otherwise can't hold the model, at a latency cost that the
economics (#23) say only pays off for a fast quantized engine at high volume.
