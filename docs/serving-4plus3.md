# Serving blueprint: 4+3 pods, Qwen3.6-35B-A3B Q4_K, 7× Vega 56

Target: two independent llama-server pods on disjoint cards, every stream at (or near)
128k context, zero CPU inference, minimal host RAM. Assumes the 16 GB RAM upgrade —
measured fact: two instances on 3.8 GB die to OOM on pinned GTT no matter what
(zram cannot swap pinned pages; see ledger 2026-08-01).

## Why 4+3 and not one 7-card server

Slots inside one server share a single compute budget: prefill saturates at ~445-470
aggregate no matter how many slots, and per-stream decode is aggregate/streams.
Independent pods have independent compute, so aggregate prefill ≈ the sum of pods,
and fewer boundaries per pod beat more GPUs per pod at equal load
(4-card 471 PP / 52.2 TG vs 6-card 425 / 48.6, ledger 2026-08-01).

## Pod layout (VRAM math, q8_0 KV, 1.38 GiB per 128k stream incl. GDN state)

| pod | cards | weights | KV budget | streams |
|---|---|---|---|---|
| A | 4 (31.9 GiB) | 21.3 GiB | ~8.5 GiB after buffers | **4×128k** comfortable; 6×128k aggressive; 8×128k does NOT fit (measured) |
| B | 3 (23.9 GiB) | 21.3 GiB | ~2 GiB after buffers | **1×128k + margin**; try 2×96k, fall back if alloc fails |

Total: 5-6 streams at 128k-class context ≥ the 4×128k / 512k goal.

## Launch (validated flags; conservative no-spill policy)

```bash
# Pod A - 4 cards
GGML_VK_VISIBLE_DEVICES=0,1,2,3 RADV_PERFTEST=nogttspill \
  vega-runtime/bin/llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -c $((4*131072)) -np 4 --host 0.0.0.0 --port 8080

# Pod B - 3 cards
GGML_VK_VISIBLE_DEVICES=4,5,6 RADV_PERFTEST=nogttspill \
  vega-runtime/bin/llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -c 131072 -np 1 --host 0.0.0.0 --port 8081
```

Front with any OpenAI-compatible balancer (nginx `least_conn` across :8080/:8081).
Route long-context/batch work to pod A, latency-sensitive single streams to pod B
(3-card pod has the best single-stream prefill: 565-675 t/s).

## RAM-minimal rules

- Keep default mmap (do NOT use `--no-mmap` in serving): weight pages are read-only,
  shared between both pods, and evictable after VRAM upload. No `--mlock`.
- q8_0 KV is mandatory at 128k (21.0 vs 9.0 t/s at f16 — spill cliff), and KV lives in VRAM.
- Steady-state host footprint ≈ 1-1.5 GiB pinned GTT staging per pod + ~1 GiB anon per
  pod: ~5 GiB total on a 16 GB box, rest = page cache (makes restarts ~4× faster).
- Never any CPU offload: `-ngl 99` and no `--override-tensor` (tensor overrides also
  silently disable pipeline parallelism).

## Expected numbers (from measured baselines, b9599 + patch)

- Aggregate prefill both pods busy: ~1000-1150 t/s. Do **not** set the Bonsai production
  `GGML_VK_TILE_M` value for this Qwen Q4_K model: although it appeared 19% faster and kept
  a long greedy prefix, perplexity exploded from 131.03 to 3.58 million.
- Decode: pod A 52 t/s aggregate at 4 active streams (32 solo); pod B 31 solo.
- Single-stream decode tuning: `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` is the measured winner
  (29.44 → 34.87 t/s, byte-identical output). It does **not** stack additively with the
  conservative `RADV_PERFTEST=nogttspill` policy: graphics + nogttspill gives 32.08 t/s,
  and adding the async transfer queue gives 33.78 t/s. Keep `nogttspill` in the launch
  commands so an over-capacity configuration fails instead of silently paging weights over
  Gen2 x1; use graphics-queue alone only after the exact production slot/context allocation
  is proven to fit.
- `expert_used_count` 8→6 improves the 5.6k-prompt probe to 559 PP / 31.1 TG, but changes
  model semantics and remains experimental until perplexity and task-quality evaluation.
- MTP self-speculation is **not** a pending multiplier on this rig: it was 7.6× slower at
  79% acceptance because every draft step pays device-to-host traffic over Gen2 x1.
