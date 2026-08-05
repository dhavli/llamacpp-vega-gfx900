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
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
LLAMA_SERVER_FULL_OUTPUT_RESERVE=1 \
  vega-runtime/bin/llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 \
  -c $((4*131072)) -np 4 --cache-ram 0 --ctx-checkpoints 0 \
  --host 0.0.0.0 --port 8080

# Pod B - 3 cards
GGML_VK_VISIBLE_DEVICES=4,5,6 RADV_PERFTEST=nogttspill \
LLAMA_SERVER_FULL_OUTPUT_RESERVE=1 \
  vega-runtime/bin/llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
  -c 131072 -np 1 --cache-ram 0 --ctx-checkpoints 0 \
  --host 0.0.0.0 --port 8081
```

Front with any OpenAI-compatible balancer (nginx `least_conn` across :8080/:8081).
Route long-context/batch work to pod A, latency-sensitive single streams to pod B
(3-card pod has the best single-stream prefill: 565-675 t/s).

If the **500 PP / 25 TG per-stream floor** is an SLA, the balancer must admit at most one
active generation per pod. Four simultaneous requests on pod A completed, but individual
streams measured only 130-492 PP and 3.4-13.0 TG because they share one compute pipeline.
The four slots are context-capacity slots, not four independent ≥25 TG engines. Queue excess
work or run it in best-effort capacity mode; two pods provide two concurrent SLA streams.

## RAM-minimal rules

- On the current 3.8 GiB host, use `--no-mmap` for a single four-card pod: the exact 4x128k
  allocation was OOM-killed during upload with mmap. Re-evaluate default mmap after the 16 GiB
  upgrade, where its read-only pages can be shared between both pods. Never use `--mlock`.
- q8_0 KV is mandatory at 128k (21.0 vs 9.0 t/s at f16 — spill cliff), and KV lives in VRAM.
- Steady-state host footprint ≈ 1-1.5 GiB pinned GTT staging per pod + ~1 GiB anon per
  pod: ~5 GiB total on a 16 GB box, rest = page cache (makes restarts ~4× faster).
- Never any CPU offload: `-ngl 99` and no `--override-tensor` (tensor overrides also
  silently disable pipeline parallelism).
- `LLAMA_SERVER_FULL_OUTPUT_RESERVE=1` is mandatory with this runtime. Without it, server's
  one-slot graph reservation decodes at 3.77 t/s; with it, the same request reaches 29.70,
  near `llama-completion` at 32.47. `--cache-ram 0 --ctx-checkpoints 0` avoids accumulating
  ~129 MiB prompt entries and ~62.8 MiB recurrent checkpoints on the RAM-minimal setup.

## Expected numbers (from measured baselines, b9599 + patch)

- Aggregate prefill both pods busy: ~1000-1150 t/s. Do **not** set the Bonsai production
  `GGML_VK_TILE_M` value for this Qwen Q4_K model: although it appeared 19% faster and kept
  a long greedy prefix, perplexity exploded from 131.03 to 3.58 million.
- Decode: pod A 52 t/s aggregate at 4 active streams; the exact four-slot/single-admission server
  reaches 26.95 TG with the graphics queue. Pod B historically reached about 31 solo.
- The 52 t/s batched-bench figure is aggregate, not per-stream. A real four-request server
  run produced 3.4-13.0 TG per stream depending on scheduler interleaving. Solo mode is the
  only measured configuration that meets ≥25 TG per active stream.
- Single-stream decode tuning: `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` is the measured winner
  (29.44 → 34.87 t/s historically; 30.65 → 35.47 in the latest production-prompt A/B,
  byte-identical output). Pair it with `-ts 0.85,1.05,1.05,1.05` on the forward-ordered
  four-card pod: PP improves 529.67 → 559.74 while TG holds 35.56. It does **not** stack additively with the
  conservative `RADV_PERFTEST=nogttspill` policy: graphics + nogttspill gives 32.08 t/s,
  and adding the async transfer queue gives 33.78 t/s. Keep `nogttspill` in the launch
  commands so an over-capacity configuration fails instead of silently paging weights over
  Gen2 x1. The exact 4x128k allocation now passes with both `nogttspill` and graphics queue,
  automatic placement, and ub512: 529.63 PP / 26.95 TG versus 529.45 / 23.41 without graphics,
  with byte-identical output. This meets the 500/25 single-admission SLA.
  Do not add `GGML_VK_PER_QUEUE_MUTEX=1` to this long-context profile: it keeps PP flat at
  529.66 but lowers TG to 26.02. The per-device-lock win is specific to the shallow profile.
  For the opt-in reduced-expert service tier, top-6 is the exact-shape winner: repeated runs
  measured 573.16/29.03 and 571.35/27.58 PP/TG (572.26/28.31 average), with deterministic
  same-tier output. Top-7 measured 551.76/25.78 and is a mixed tradeoff here.
- Optional quality/speed tradeoff: `--override-kv qwen35moe.expert_used_count=int:6`
  improves the 5.6k-prompt probe to 559 PP / 31.1 TG. PPL changed only 131.03 → 131.42 and
  three deterministic tasks remained correct, but this is still a narrow evaluation;
  top-8 remains the production default.
- A final-runtime top-7 tier (`expert_used_count=int:7`) is the better latency/decode tradeoff:
  609.39 PP / 36.04 TG and PPL 132.3556 +/- 5.2767. Top-6 is prefill-biased at
  629.80 / 33.94. Both matched or exceeded top-8 on the bounded repeated eight-task suite,
  but both change trained routing semantics; expose them as explicit opt-in service tiers.
  That top-7 preference applies to the shallow completion profile only. With four 128k slots
  allocated, top-6 dominates top-7 and top-8 on both PP and TG.
- At the exact 4×128k allocation, graphics queue changed four-request wall time only
  53.29 → 52.60 seconds (~1%). Keep the conservative `nogttspill` launch default; the
  graphics-queue solo win is not a meaningful mixed-concurrency win.
- MTP self-speculation is **not** a pending multiplier on this rig: it was 7.6× slower at
  79% acceptance because every draft step pays device-to-host traffic over Gen2 x1.
