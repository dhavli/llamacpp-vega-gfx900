# Qwen3.6 four-card optimization roadmap

Target: Qwen3.6-35B-A3B UD-Q4_K_XL, top-8, four forward-ordered Vega devices
(`GGML_VK_VISIBLE_DEVICES=0,1,2,3`), zero CPU inference.

## Best validated shallow-context profile

```bash
GGML_VK_VISIBLE_DEVICES=0,1,2,3 \
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
LLAMA_SERVER_FULL_OUTPUT_RESERVE=1 \
  vega-runtime/bin/llama-server \
  -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on -ts 0.85,1.05,1.05,1.05 \
  -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 \
  --cache-ram 0 --ctx-checkpoints 0
```

The production-shaped 5,629-token completion A/B measured:

| profile | PP | TG | output |
|---|---:|---:|---|
| default queue/split, 800 HBM | 528.72 | 30.65 | oracle |
| graphics queue, default split | 529.67 | 35.47 | identical |
| graphics queue, split `0.85,1.05,1.05,1.05` | **559.74** | **35.56** | identical |
| graphics queue, split `0.70,1.10,1.10,1.10` | 560.09 | 33.36 | identical |
| fast split, top-6 experts | **608.61** | 33.66 | semantics changed |

Use the first custom split for top-8: it adds 5.7% PP without losing the graphics-queue TG
gain. Top-6 remains an experimental quality/speed tier, not the production default.

Hard configuration findings:

- Preserve forward device order. Reversing it produced only 88.55 PP / 2.81 TG.
- Keep `-ub 512`. `-ub 1024` became pathologically slow and was terminated after several
  post-load minutes; `-ub 256` had already lost 13.6% PP.
- Keep stock 1590 MHz core and 800 MHz HBM for four-card serving. HBM 900 changed PP/TG by
  only +0.36%/+0.36% on the default queue, and under graphics queue changed +0.5%/-1.8%.
- Do not combine the peak profile with `RADV_PERFTEST=nogttspill` until the exact allocation
  has passed a fit gate; that conservative policy reduces the graphics-queue solo win.

## Ranked implementation work

1. **Completed:** an opt-in pipeline-safe scheduler tracer now records split counts, boundary
   bytes, event waits, asynchronous and fallback-copy time, source/destination synchronization,
   submit time, and final synchronization without enabling Vulkan's pipeline-hostile perf logger.
   On the 5,629-token prefill, each downstream boundary moved 46,129,200 bytes in 24 fallback
   copies; the three staged-copy legs consumed about 0.8 seconds of a 10.1-second prefill. A
   direct coherent mapped-memory copy was tested and rejected: uncached/BAR reads inflated copy
   time to roughly 11.7 seconds and collapsed PP to 269.31. Dedicated transfer queues were
   neutral. A blocking two-slot, 1 MiB double-buffered pipeline was then implemented and traced.
   It overlapped 333 ms while moving 138 MB, but same-binary ON/OFF was 568.59/34.40 versus
   560.11/35.57 PP/TG, with only a 0.16% total-request improvement. It was removed because the
   +1.5% PP signal misses the 2% gate and TG regresses 3.3%. Keep the stock staged-copy path.
2. **Completed:** synchronization-free fixed-counter histograms show that real prefill Q4_K
   dispatches are exclusively in the 257–512-token MM bucket (198–242 calls per card). The
   only vec calls are the 2–8-token final/decode tail. This removes a vec/MM cutoff sweep from
   the prefill candidate list.
3. **Completed / neutral:** deterministic per-expert row-ID precompaction was implemented behind
   an opt-in path. A portable shared-prefix preparation was byte-identical but regressed to
   549.02 PP. A subgroup-ballot refinement was also byte-identical and recovered 559.22 PP,
   only +0.14% over the 558.42 traced control. The patch and its ~4 MiB/card ubatch scratch were
   removed because they do not clear the adoption threshold.
4. **Completed / rejected:** upstream PR #24720 command-buffer graph reuse was ported default-off
   with lifecycle counters and unsafe timed eviction removed. It captured twice and replayed 123
   decode graphs per card, but same-binary ON/OFF measured 559.38/33.81 versus 559.23/35.46 and
   the replayed output diverged into repeated multilingual garbage. The full series was removed.
   Do not retry without first fixing its stale/dynamic graph-state correctness model.
5. **Completed / rejected:** the post-26.1.5 ACO single-wave workgroup-barrier elimination plus
   its required correctness fix was built as an isolated Mesa 26.1.5 runtime. Output remained
   byte-identical, but performance collapsed from 559.23/35.46 to 511.27/31.31 PP/TG. The Mesa
   override was removed.
6. **Completed / rejected:** the complete gfx9-applicable RADV thread-ID/shuffle series through
   `f44a6b57` was adapted to Mesa 26.1.5 and built separately. Output was byte-identical, but
   performance fell to 507.49/31.89 PP/TG (-9.3%/-10.1%). The override was removed. The remaining
   UUB pass affects under 1% of general shaders and is not credible without matching active IR.
7. After the 16 GB host-RAM upgrade, revisit the already-pinned tensor-parallel path and
   experimental Vulkan AllReduce. Do not attempt it on the current 3.8 GB host.

## Upstream audit conclusions

- PR #25862 (`5cea9089`) exactly matches 256-expert MoE routing, but the four-card A/B was
  neutral: 528.56 PP versus 528.72 control. It was removed.
- Post-pin top-k fusion PR #26124 targets `sqrt(softplus)`, not Qwen3.6's already-active
  `TOPK_MOE_EARLY_SOFTMAX_NORM`; expected benefit is zero.
- Eye-catching Qwen prefill patches #25483 and #22970 require cooperative matrices, which
  gfx900 does not expose. NonTemporal streamed-weight work targets GFX10+, not Vega10.
- The post-Mesa-26.1.5 RADV thread-ID/subgroup series was isolated and tested; it regressed
  PP/TG by 9.3%/10.1% and was removed. The remaining UUB pass has no matching active-shader IR
  evidence and an estimated sub-1% general-shader reach, so it is excluded from implementation.

Every implementation candidate must pass deterministic output, RAM-safe perplexity when
math/order changes, no CPU placement/spill/faults, and repeated PP/TG gates before adoption.
