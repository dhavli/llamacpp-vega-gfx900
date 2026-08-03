# Qwen3.6 four-card optimization roadmap

Target: Qwen3.6-35B-A3B UD-Q4_K_XL, top-8, four forward-ordered Vega devices
(`GGML_VK_VISIBLE_DEVICES=0,1,2,3`), zero CPU inference.

## Best validated shallow-context profile

```bash
GGML_VK_VISIBLE_DEVICES=0,1,2,3 \
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
GGML_VK_PER_QUEUE_MUTEX=1 \
LLAMA_SERVER_FULL_OUTPUT_RESERVE=1 \
  vega-runtime/bin/llama-server \
  -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on -ts 0.85,1.05,1.05,1.05 \
  -ctk q8_0 -ctv q8_0 -b 2048 -ub 768 \
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
- Use `-b 2048 -ub 768` for this four-card shallow-context profile. Repeated runs measured
  583.34–584.80 PP and 34.93–35.64 TG, with deterministic output. Streaming PPL was 131.2037
  versus 131.0328 control. Ubatch 704 was slightly slower; 832, 896, and 1024 were pathological.
  Ubatch 704's 582.13/35.59 and 13.467-second total was a noise-level Pareto tradeoff, not a
  >=2% win. The aligned `-b2304 -ub768` pair regressed TG to 32.66, worsened total time to
  13.746 seconds, and changed the output hash, so keep batch 2048.
- Keep stock 1590 MHz core and 800 MHz HBM for four-card serving. HBM 900 changed PP/TG by
  only +0.36%/+0.36% on the default queue, and under graphics queue changed +0.5%/-1.8%.
- `RADV_PERFTEST=nogttspill` is optional diagnostic hardening, not a production speed knob. The
  exact peak profile passed it at 585.54 PP / 34.69 TG with 42/42 layers offloaded, no evictions,
  invalidations, or GPU faults; its performance effect is neutral.

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
   A v2 with one persistent writer per tensor (rather than per chunk) also failed: 1 MiB measured
   569.59/32.11 with 645 ms copy wall, while 512 KiB measured 568.53 PP and 660 ms wall. Both are
   no better than v1's 639 ms wall; larger chunks reduce overlap, so the bounded sweep was closed.
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
8. **Completed / retained:** the custom split maps to 9/11/11/9 transformer blocks with the
   output layer on card 3. Its six one-block neighbors were exhaustively screened. Five measured
   528.47–529.94 PP; the least-bad measured 558.47/32.83 PP/TG. All were deterministic, so the
   current allocation is a one-block local optimum.
9. **Completed / rejected:** perf attribution showed Q4_K medium `MUL_MAT_ID` is 27.72% of
   instrumented prefill GPU time, an unswept surface distinct from prior TILE_S/TILE_M tests.
   Valid BN128 and BM128 geometries regressed to 462.18 and 406.14 PP. A 128-thread/BM32 tile
   reached 626.14 PP but diverged and produced PPL 248,320 versus 131.03 control. The knob was
   removed; the apparent win was invalid work partitioning.
10. **Completed / adopted:** the bounded upper ubatch bracket found 768 as the quality-safe
    optimum. Repeats measured 583.34–584.80 PP and 34.93–35.64 TG; streaming PPL measured
    131.2037 versus 131.0328 control. Ubatch 704 was lower, while 832/896 crossed the pathological
    cliff. Batch sweeps at 2560 and 3200 were mixed or worse, and aligned b2304/ub768 regressed
    decode and total time. The final `nogttspill` run offloaded 42/42 layers with no eviction or
    faults. Ubatch 768 with batch 2048 is the closed-matrix shallow-context winner.
11. **Completed / rejected:** upstream PR #25005's FLOP-based graph-submission heuristic was
    backported behind an opt-in A/B. It preserved output but changed 583.75/32.00 PP/TG and
    13.855 seconds to 574.56/35.28 and 13.639 seconds. The 1.57% PP regression and only 1.56%
    total improvement miss the gate and agree with the later upstream MoE regression report.
    The patch was removed.
12. **Completed / adopted:** the process-global Vulkan submission mutex was narrowed to one
    mutex per underlying device queue, while aliased compute/transfer wrappers still share a
    lock. Two-run global-lock averages were 583.95/32.77 PP/TG and 13.761 seconds; per-device
    locks averaged 584.71/34.64 and 13.531 seconds (+0.13% PP, +5.7% TG). The fail-closed run
    reached 583.74/35.56 with identical output and no faults. Export
    `GGML_VK_PER_QUEUE_MUTEX=1`.

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
- Upstream PR #22933's GCN subgroup reduction was tested through the existing
  `GGML_VK_GCN_SUBGROUP_REDUCE=1` gate. Same-binary OFF/ON was 561.10/35.48 versus
  561.02/35.23 PP/TG, byte-identical. The gfx900-inapplicable integer-dot vecq half was not ported.

Every implementation candidate must pass deterministic output, RAM-safe perplexity when
math/order changes, no CPU placement/spill/faults, and repeated PP/TG gates before adoption.
