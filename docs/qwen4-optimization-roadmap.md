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
   neutral. Keep staged copies until a genuinely asynchronous/double-buffered design is ready.
2. **Completed:** synchronization-free fixed-counter histograms show that real prefill Q4_K
   dispatches are exclusively in the 257–512-token MM bucket (198–242 calls per card). The
   only vec calls are the 2–8-token final/decode tail. This removes a vec/MM cutoff sweep from
   the prefill candidate list.
3. **Completed / neutral:** deterministic per-expert row-ID precompaction was implemented behind
   an opt-in path. A portable shared-prefix preparation was byte-identical but regressed to
   549.02 PP. A subgroup-ballot refinement was also byte-identical and recovered 559.22 PP,
   only +0.14% over the 558.42 traced control. The patch and its ~4 MiB/card ubatch scratch were
   removed because they do not clear the adoption threshold.
4. **Next:** test Vulkan command-buffer graph reuse from upstream PR #24720 behind an opt-in build.
   The 2-core Celeron makes submission overhead plausible, but upstream reports no measured
   gain; require at least 2% TG plus multi-turn server correctness.
5. After the 16 GB host-RAM upgrade, revisit the already-pinned tensor-parallel path and
   experimental Vulkan AllReduce. Do not attempt it on the current 3.8 GB host.

## Upstream audit conclusions

- PR #25862 (`5cea9089`) exactly matches 256-expert MoE routing, but the four-card A/B was
  neutral: 528.56 PP versus 528.72 control. It was removed.
- Post-pin top-k fusion PR #26124 targets `sqrt(softplus)`, not Qwen3.6's already-active
  `TOPK_MOE_EARLY_SOFTMAX_NORM`; expected benefit is zero.
- Eye-catching Qwen prefill patches #25483 and #22970 require cooperative matrices, which
  gfx900 does not expose. NonTemporal streamed-weight work targets GFX10+, not Vega10.
- A small post-Mesa-26.1.5 RADV thread-ID/subgroup series is a low-priority <=1% candidate;
  isolate it from other Mesa changes if tested.

Every implementation candidate must pass deterministic output, RAM-safe perplexity when
math/order changes, no CPU placement/spill/faults, and repeated PP/TG gates before adoption.
