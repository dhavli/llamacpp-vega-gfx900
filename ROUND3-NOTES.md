# Round-3 work notes (toward TG 30 / PP "bigger the better")

State after round 2 (commit 1e6b91f, runtime store `pk52zfmn...`): TG 18.65, pp512 141.1.
Decode budget 53.6 ms/token = ~33 ms matvec + ~15-20 ms small-op dispatch swarm.

## (a) TG: fuse delta-net elementwise chains — SIGMOID+MUL DONE (perf-neutral, v9)
The scan-based matcher + sigglu shader landed; fires 16×/token (attention gates only),
bit-identical, neutral timing. The remaining dispatch swarm is the 48 delta-net layers:
GET_ROWS×2/SCALE/CPY/L2_NORM×2/CONCAT per layer — requires GDN-integrated state indexing
(pass ids into GATED_DELTA_NET and index inside; graph + shader change).
Sharpened diagnosis (2026-07-31): the ~15 ms/token small-op cost is BARRIER DRAINS on the
dependent per-layer chain (~500-700 barriers x ~20 us GCN pipeline drain), not dispatch
issue cost — zero-size ops are already skipped (ggml_vk_is_empty), sync elision already
skips barriers between independent ops, and graph reordering cannot help a sequential
dependent chain. Only fusing chain links helps: each fused pair removes one barrier.
Ranked fusion candidates by links removed per layer (48 layers): state GET_ROWS into
SSM_CONV/GDN (2), CPY/SET_ROWS state writeback into GDN (2), L2_NORM pair into GDN
prologue (2), SCALE/SOFTPLUS/SIGMOID into GDN prologue (2-3). A GDN megakernel absorbing
its elementwise prologue+epilogue could remove ~6-8 barriers/layer = ~6-8 ms/token -> ~24 t/s.
Implementation entry points (ggml-vulkan.cpp, prism clone):
- Fusion detection chain: ~line 15808-15861 (`ggml_vk_can_fuse(ctx, cgraph, i, {...})`
  else-if ladder inside graph build) — add `{GGML_OP_UNARY(SIGMOID), GGML_OP_MUL}` case here.
- Matcher template: `ggml_vk_can_fuse_ssm_conv` at :15331 (checks ggml_can_fuse + type/
  contiguity/misalign guards; require sigmoid-out consumed only by the mul, same shapes).
- Generic matcher: `ggml_vk_can_fuse` at :15177.
- Dispatch uses `ctx->num_additional_fused_ops` + dst = `cgraph->nodes[node_idx + n_fused]`
  (see RMS_NORM_MUL dispatch ~:8960 for the two-tensor fused pattern).
- Shader: new `sigmoid_mul.comp` (elementwise dst = sigmoid(a)*b), or extend glu.comp.
- The out-ids GET_ROWS at n==1 is ALREADY skipped by the fork (guard in qwen35.cpp:216 +
  llama-graph.cpp:242) — the 97 GET_ROWS/token are all state-cache reads (GDN-with-ids
  fusion needed, heavier).
Source of truth: `src/models/qwen35.cpp` in the prism clone (scratchpad) — per layer:
- `ggml_mul(cur, ggml_sigmoid(ctx0, gate))` at ~line 708 → SIGMOID+MUL fusion (pattern like
  RMS_NORM_MUL; matcher framework at `ggml_vk_can_fuse` ggml-vulkan.cpp:15177, examples:
  `ggml_vk_can_fuse_ssm_conv` :15331 for SSM_CONV(+ADD)+UNARY).
- L2_NORM ×2 (q_conv/k_conv, lines 536-537) — could fuse into one dispatch or into
  neighboring ops.
- `inp_out_ids` GET_ROWS (lines 217/218/290/739): at n==1 these are identity gathers —
  check whether upstream's "skip when n_outputs==n_tokens" logic is missing in this fork.
- GET_ROWS ×2/layer for delta-net state cache read + SET_ROWS write — fusing into
  GATED_DELTA_NET (pass ids, index inside) is the big one but needs graph + shader changes.
- Note: GGML_SCHED_DEBUG=2 prints nothing in the release build; to get the exact node
  sequence use a debug build or add a graph-print in ggml_vk_graph_compute.

## (b) PP: mul_mm LDS double-buffering — DONE, FALSIFIED (patch v10)
Implemented as GGML_VK_MM_DBUF (2× LDS tiles, one barrier/tile): 133.1 vs 138.0 at the
128×64 tile — the doubled shared memory halves WG occupancy per CU, costing more than the
saved barrier drains. Correct output; knob retained. mul_mm micro-opt space is exhausted at
pp512 141.1; further PP would need persistent kernels or int8-B with v_mad_i32_i24 emulation.

## (c) TG: generic n=2..8 dmmv is slow (740 µs vs 113 µs n=1 for 8× work)
The SoA generic path (dequantize/get_dm per 4-8 weights) is byte-era code; a wide-style
NUM_COLS>1 kernel with the LUT trick could ~2× short-prompt latency. Matters for
interactive use, not for pp512.

## Consultants (user request, blocked on auth)
Brief ready: session scratchpad `consult-brief.md` (also usable standalone). `codex` needs
`codex login` (401), `agy` needs Google OAuth (`agy` interactive once). Re-run:
`agy --print --dangerously-skip-permissions --effort high "$(cat consult-brief.md)"` and
`codex exec --dangerously-bypass-approvals-and-sandbox "$(cat consult-brief.md)"`.

## Falsified this session (do not retry; details RESULTS.md §5e)
fp16 staging alone; occupancy alone; SoA coalescing alone (matvec is ALU-bound);
LOAD_VEC_A=16 load_a; BK_STEP=4 V4 LDS loads; TERNARY_BS 128/256; TERNARY_ROWS ≠ 8;
mm tiles: BK≠32, TM/TN=8, tiles >128×128, BN>64 with packed-k accumulators.

## GDN rows-mode Vulkan port (in progress 2026-07-31 ~05:00)
DISCOVERY: the state fusion already exists as `ggml_gated_delta_net_rows` (ggml.h:2565) —
reads per-seq state from the 2D cache view at rows[seq]; kills per-layer gather + slot-0 cpy.
Implemented on CPU (+Metal); Vulkan rejects src[6] in supports_op (ggml-vulkan.cpp:~17234)
and qwen35.cpp (~line 470) additionally requires all-GPU-devices==Metal to emit it.
CPU reference (ggml-cpu/ops.cpp:~10565): ONLY input addressing differs:
  s_in = state_base + rows[seq]*(state.nb[1]/4) + head*S_v*S_v   (vs seq*K*H*D + head*D)
Output layout unchanged (attn | K snapshot slots).
Vulkan port pieces:
1. Shader gated_delta_net.comp: `#if GDN_ROWS` → binding 7 = I32 rows[]; push const
   `state_row_stride`; state_in_base = rows[seq_id]*state_row_stride + head_id*state_size.
   (state_in_base currently `(seq_id*K*H + head_id)*state_size`, rel line ~107.)
2. In rows mode K = n_snap_slots comes from op_params (VERIFY in ggml.c ctor), NOT
   src_state->ne[1] (that's the cache row count) — fix host dispatch K computation.
3. Pipelines: mirror pipeline_gated_delta_net[3][2] (created ggml-vulkan.cpp:5351, selected
   :10683) as _rows variant, 8 bindings, gen-emitted with {"GDN_ROWS","1"}.
4. Dispatch ggml_vk_gated_delta_net (~11552): if src[6]: rows pipeline, bind rows buf,
   pass state_row_stride = src_state->nb[1]/4.
5. supports_op: accept src[6] type I32.
6. qwen35.cpp device check: accept reg_name "Vulkan" too.
Validate: GGML_GDN_STATE_GATHER=1 (legacy) vs rows-mode A/B, md5 + ab-bench.

## GDN rows-mode port — DONE, correct; FALSIFIED as TG lever (v11, commit 9890a98)
Vulkan port works (md5-identical in all modes; LLAMA_N_RS_SEQ env override added for
testing). But rows mode only engages with the rollback ring (n_rs_seq>0), whose K snapshot
slots make GDN write per-token state snapshots — 4.0 t/s vs 18.6. The pure fusion win needs
a NEW graph formulation: rows-read + single in-place state write WITHOUT snapshot slots
(i.e. ggml_gated_delta_net_rows with n_snap_slots decoupled from the ring, or an in-place
state-write variant). That plus absorbing the GDN elementwise prologue (sigmoid/softplus/
scale/L2_NORM) is the remaining TG path (~24 t/s est.) — novel design work, good candidate
to sanity-check with the consultants once authenticated.

UPDATE (v11.3, commit 6723976): the 4.0 t/s was NOT snapshot traffic — the ring conv path
fed a strided view into im2col; Vulkan requires contiguous src1 → ~48 graph splits to the
Celeron. Fixed with ggml_cont (delta-net-base.cpp): rows-active now 13.3 t/s all-GPU,
md5-identical, zero rejections. Gathered (18.6) remains the decode default — the ring
snapshot ops (~21 ms/token) outweigh the saved gathers. New diagnostic:
GGML_VK_SOA_DEBUG=1 logs every supports_op rejection (how this was found).

---

# Round-4 (2026-07-31): critical-path correction + consultant round

## THE methodological correction — rank by critical path, not perf-logger total
Implemented the snapshot-free K=1 GDN rows path (`build_recurrent_attn`: when
`n_rs_seq == 0` and a backend supports rows mode, read state via `state_rows`
instead of gather + slot-0 cpy; gate in qwen35.cpp no longer requires
`n_rs_seq > 0`; `GGML_GDN_STATE_GATHER=1` restores the old path).

Result: GET_ROWS 97 ops / 3.31 ms  ->  49 ops / 0.86 ms. Output BIT-IDENTICAL
(md5 9e30dd54e452). GPU op time saved: 2.45 ms.
Wall clock: 18.53 -> 18.66 t/s (53.96 -> 53.59 ms). **Only 0.37 ms.**

Why: the state gather reads the KV cache and does NOT depend on the current
layer's activations, so it was already overlapping with matvec execution. The
perf logger serializes dispatches and therefore bills off-critical-path ops at
full cost. **Do not size future work off GGML_VK_PERF_LOGGER totals.** Only ops
on the dependent chain are worth removing.

Kept anyway: bit-identical, small TG win, and a solid short-prefill win
(65-token prompt eval 51.0 -> 60.3 t/s).

## Decode is 100% GPU-bound
Over 10 s of steady decode: 0 ms utime, 0 ms stime, 1 thread (command buffers
reused, "graphs reused = 126"). The 2-core Celeron is NOT a decode bottleneck.
Stop considering submission overhead.

## Full per-token GPU profile (v11.3 + best env, GGML_VK_PERF_LOGGER=1, ~60 ms instrumented)
Matvec (all q1_0, all on the critical path): ~39.9 ms total
  m=17408 k=5120   128x 122.8 us = 15.72 ms   <- 12.5 MB in 122.8 us = 102 GB/s
  m=5120  k=17408   64x 118.9 us =  7.61 ms   (MUL_MAT_ADD)
  m=10240 k=5120    48x  82.7 us =  3.97 ms
  m=6144  k=5120    48x  64.1 us =  3.08 ms
  m=5120  k=6144    48x  51.1 us =  2.45 ms
  m=48    k=5120    96x  23.8 us =  2.29 ms   <- alpha/beta projections, 20 GFLOPS, ~pure overhead
  m=248320 (output head) 1x 1.22 ms
Non-matvec: RMS_NORM_MUL 4.42, GET_ROWS 3.31 (now 0.86), CPY 2.08,
  GATED_DELTA_NET 1.75, GLU 1.43, ADD 1.29, MUL 1.19, L2_NORM 1.15,
  FLASH_ATTN 1.08, CONCAT 0.63, SOFTPLUS 0.63, SIGMOID 0.62, SILU 0.61,
  ROPE 0.41, SET_ROWS 0.39, CONT 0.22

## Matvec is LATENCY-bound, not ALU-bound (revises round-2 conclusion)
102 GB/s = 25% of the 410 GB/s roofline. Instruction accounting for the LUT inner
loop (per 4 weights: 1 bfe + 1 ds_read_b64 + 2 pk_fma) allows ~878 GB/s of weight
traffic on VALU alone, so VALU is not binding. At 64 VGPR / 4-of-10 waves the wave
issues one row's load, hits s_waitcnt, and eats a full ~350-cycle HBM latency with
nothing to backfill.

Rows-per-WG is saturated (sweep, bs=64): rows=4 -> 14.93, rows=8 -> 18.14,
rows=12 -> 17.91, rows=16 -> 18.31. bs=128 uniformly worse (16.58 at rows=8) and
changes the reduction order (different md5). rm_ter is capped at 16 in the host code.

Next: `GGML_VK_TERNARY_PREFETCH=1` (new `widelutpf` shader variant) hoists all
NUM_ROWS (weight-word, scale) loads ahead of the FMA chains -> 2*NUM_ROWS loads in
flight before the first wait. Both consultants independently recommended exactly this.

## Consultant round (codex = gpt-5.6-sol; agy = Antigravity)
Codex: careful and broadly correct; agreed on latency-bound diagnosis; staged
bounds 20-25 t/s solid / 25-28 very good / 30 excellent / 35+ unlikely. Its novel
idea is a T-MAC-style **activation-side partial-sum LUT**: per 128-weight block
split activations into 32 groups of 4, precompute L_g[p] = sum of +-x over the 16
sign patterns (8 with the L_g[p^15] = -L_g[p] symmetry), then the weight nibble
indexes the table: y_r += d_r * sum_g L_g[q_rg]. ~16 packed adds per 128 weights
instead of ~64 packed FMAs, and the scale applies once per block. Needs >=16 rows
per WG to amortize LUT construction. Also: try PP double-buffering at EQUAL LDS
footprint (BK=32 x 2 buffers vs BK=64 x 1) — our falsification doubled LDS and lost
to occupancy, which does not falsify the idea.

agy: more optimistic and partly wrong. Its headline recommendation — pre-format
sign bits at bit 15/31 of a dword at repack time so the loop is v_xor_b32 +
v_pk_add_f16 with zero LDS — is UNSOUND: a 32-bit mask covering 2 weights means
16 bits of storage per weight, a 16x expansion that destroys the 1.125 bpw
bandwidth advantage. Codex analyzed the same idea and rejected it correctly.
agy's claimed ceilings (65-70 t/s, 335 GB/s matvec) are not credible.

Full transcripts: scratchpad/codex-consult.out, scratchpad/agy-consult.out.
Brief (updated to current state): scratchpad/consult-brief.md.

## Ranked remaining work
1. Matvec memory-level parallelism (prefetch variant built; then multiple partial
   accumulators per row, and per-shape rows/WG selection).
2. T-MAC activation-side LUT matvec at 16-32 rows/WG.
3. GDN prologue fusion — but ONLY the links actually on the dependent chain
   (sigmoid/softplus/add/mul on alpha+beta, L2_NORM pair). Re-estimate honestly:
   the GET_ROWS result shows adjacency does not imply removable cost.
4. Merge the alpha+beta projections (m=48 each, 96 dispatches, 2.29 ms at 20 GFLOPS).
5. PP: equal-LDS double buffering (BK=32 x2).

## Round-4 results (verified, 2026-07-31)

Two code changes + one hardware finding took TG 18.67 -> 21.67 and pp512 141 -> 160.5.

| change | TG | pp512 | notes |
|---|---|---|---|
| session start (v11.3) | 18.67 | 141.1 | at 950 MHz core (unknown at the time) |
| + snapshot-free K=1 GDN rows | 18.66 | - | bit-identical; GET_ROWS 97->49 ops; only +0.37 ms wall (off critical path) |
| + matvec prefetch (widelutpf) | 19.48 | - | +4.1%, bit-identical, `GGML_VK_TERNARY_PREFETCH=1` |
| + core clock 950 -> 1138 MHz | 21.71 | 145.0 | +11% TG, +22% PP; free |
| + TILE_M warptile | 21.67 | 160.5 | best combined config |

Production env (all knobs):
```
GGML_VK_TERNARY_ROWS=8 GGML_VK_GCN_SUBGROUP_REDUCE=1 GGML_VK_TERNARY_SOA=1 \
GGML_VK_TERNARY_LUT=1 GGML_VK_TERNARY_PREFETCH=1 GGML_VK_MM_PACKED=1 \
GGML_VK_TILE_M=256,128,64,32,32,64,2,4,4,1,64
```
Runtime store: /nix/store/axc7pdxpxjr010a6w5i8jja3n0b2x09z-vega-runtime

### GPU clock ceiling (see memory: vega-gpu-clocks-underclocked)
The cards were pinned to 950 MHz by a HiveOS mining profile (not 1474 stock boost), so
every roofline in rounds 1-3 was computed against the wrong clock. 1138 MHz is the hard
stable ceiling: 1269 MHz hangs at both 800 and 950 mV, 1312 hangs, and `auto` (which
boosts to 1590 at 30 C) also produced ring timeouts. A hang needs a full reboot -- it
leaves unkillable D-state processes and wedged TTM kworkers that make all later
benchmarks silently worthless. Three reboots were needed this session.

### Still open
1. Matvec is latency-bound at ~102 GB/s of a 410 GB/s roofline and is 74% of the token.
   This remains the only path to 30 t/s. Next: multiple partial accumulators per row,
   cross-iteration software pipelining, per-shape rows/WG.
2. T-MAC activation-side LUT (codex's design, see above) at 16-32 rows/WG.
3. GDN prologue fusion -- but size it off the critical path, not perf-logger totals.
4. PP: equal-LDS double buffering (BK=32 x2); re-scan TILE_M at 1138 MHz since the
   compute/memory balance moved.

### Round-4b: TILE_M re-scan at 1138 MHz — BK=64 wins (+4.9% PP)
The warptile was tuned at 950 MHz; raising the clock moved the compute/memory balance
and the optimum shifted. Swept BK only (it changes K-tile depth and LDS size but not
output tiling geometry, so it cannot cause an OOB write / GPU hang):

| BK | pp512 (--mmap 0, -r 3, repeated) |
|---|---|
| 16 | 83.4 (llama-bench w/ mmap; clearly bad) |
| 32 | 160.68 / 160.58  <- previous best |
| **64** | **168.47 / 168.44** |

New best config (TG 21.56, pp512 168.38, prompt65 73.91):
```
GGML_VK_TERNARY_ROWS=8 GGML_VK_GCN_SUBGROUP_REDUCE=1 GGML_VK_TERNARY_SOA=1 \
GGML_VK_TERNARY_LUT=1 GGML_VK_TERNARY_PREFETCH=1 GGML_VK_MM_PACKED=1 \
GGML_VK_TILE_M=256,128,64,64,32,64,2,4,4,1,64
```
Warptile field order (confirmed from the code) is
`{threads, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, subgroup}`, and `warps = threads/subgroup`.
So this is 4x wave64, BM=128, BN=64, BK=64. LDS = (BM+BN)*(BK+bank)*2B ~= 27 KB at BK=64
(~14 KB at BK=32) -- both well under the 64 KB/CU budget, which **disproves agy's claim**
that the tile uses 48 KB and limits us to 1 WG/CU. Occupancy is not LDS-bound here, and
BK=64 winning means more K-depth per tile beats extra occupancy.

Measurement note: llama-bench with mmap has +-7 variance on this box; `--mmap 0 -r 3`
gives +-0.1. Use the latter for any comparison under ~5%.
Also: the ab-bench md5 captures stderr, so runs with a TILE_M banner have a different
md5 than runs without. Decode correctness is pinned by the banner-free run (9e30dd54e452);
TILE_M only affects mul_mm (prefill), not the n==1 matvec.

## Round-4c: -b / -ub (batch / micro-batch) sweep

**`-b` (logical batch) has ZERO effect on throughput. `-ub` (micro-batch) is the only knob
that matters.** `-b` only controls how many tokens are handed to llama_decode before being
split into `-ub` chunks, so it changes bookkeeping, not GEMM shapes.

Measured (1138 MHz, best env, `--mmap 0 -r 2`, so +-0.05):

| -b | ub=512 pp2048 | ub=512 pp8192 | ub=1024 pp2048 | ub=1024 pp8192 |
|---|---|---|---|---|
| 512  | 165.08 | 119.75 | - | - |
| 1024 | 165.14 | 119.82 | 168.14 | 134.74 |
| 2048 | 165.26 | 119.85 | 168.18 | 134.76 |
| 4096 | 165.22 | 119.83 | 168.08 | 134.81 |
| 8192 | 165.23 | 119.81 | 168.14 | 134.80 |

Spread across a 16x range of -b is 0.1%, i.e. noise. Don't bother tuning it.

`-ub` scaling (the win only appears on LONG prompts -- our pp512 benchmark is a single
micro-batch at ub=512 and therefore literally cannot show this effect):

| prompt | ub=512 | ub=1024 | delta |
|---|---|---|---|
| pp512  | 168.4 | 168.7 | - |
| pp1024 | 167.7 | 170.7 | +1.8% |
| pp2048 | 165.3 | 168.1 | +1.7% |
| pp4096 | 159.9 | 162.8 | +1.8% |
| pp8192 | 119.8 | 134.8 | **+12.5%** |

### -ub is VRAM-capped, not performance-capped
`-ub 2048` fails to allocate: the logits buffer is `n_vocab * ub * 4 B` =
`248320 * 2048 * 4` = 2,034,237,440 B = exactly the 2.03 GB allocation in the error.
This model's 248320-token vocab is unusually large, so each +512 of ub costs 508 MB of
VRAM. Probed ceiling on 8 GB with 3.53 GiB of weights resident: **ub=1792 fits, 2048 OOMs.**

### -ub optimum is prompt-length dependent and NON-monotonic
Full `-ub` scan at b=8192 (1138 MHz, `--mmap 0 -r 2`):

| -ub | pp2048 | pp8192 | logits VRAM |
|---|---|---|---|
| 512  | 165.2 | 119.8 | 0.51 GB |
| **1024** | **168.1** | 134.7 | 1.02 GB |
| 1536 | 166.1 | 131.2 | 1.53 GB |
| **1792** | 162.5 | **137.9** | 1.78 GB |
| 2048 | OOM   | OOM   | 2.03 GB |

Note ub=1536 is WORSE than both 1024 and 1792 at pp8192 (131.2 vs 134.7 / 137.9) --
non-monotonic, so do not assume "bigger ub is better" and interpolate; measure the points.
Likely a tiling/alignment interaction with the mul_mm tile (BM=128, BN=64).

Recommendation: **ub=1024 as the default** -- best at pp2048, near-best at pp8192, and only
1 GB of logits buffer. Use **ub=1792 only for workloads dominated by very long prompts**
(+2.4% at pp8192 but -3.4% at pp2048, and 1.78 GB of VRAM). Leave `-b` at whatever; it does
nothing.

## Round-4d: the clock ceiling was an undervolt artifact — 1590 MHz is stable

After the mining OC was cleared, the card's real stock V/F table became visible
(1138@950mV, 1269@1000, 1312@1050, 1474@1100, 1538@1150, 1590@1200mV). The mining
profile had pinned EVERY state to a flat 800 mV, which is why 1269/1312 hung earlier
and why "raising" 1269 to 900/950 mV did not help — still under the 1000 mV it needs.
**The earlier "1138 MHz is a hard ceiling / the card is degraded" conclusion was wrong.**

Clock ladder at stock voltages (all md5-identical, rings=0, fail-fast never triggered):

| DPM state | clock | TG | temp |
|---|---|---|---|
| 2 | 1138 MHz | 21.56 | 30 C |
| 3 | 1269 MHz | 22.75 | 36 C |
| 4 | 1312 MHz | 23.01 | 37 C |
| 5 | 1474 MHz | 24.62 | 38 C |
| 6 | 1538 MHz | 25.49 | 39 C |
| 7 | **1590 MHz** | **25.87** | 41 C |

Full verified bench at 1590 MHz (state 7, power cap 250 W, best env + TILE_M BK=64):

| metric | value |
|---|---|
| TG (decode, -n 128) | **26.06 t/s** |
| pp512 | **199.55** |
| pp2048 (ub=1024) | 197.03 |
| pp8192 (ub=1024) | 156.38 |
| prompt eval, 65 tok | 87.89 |
| md5 | 19d4d2c0f65e (unchanged) |
| temp | 43 C |

Session arc: TG 18.67 -> 26.06 (+40%), pp512 141 -> 199.6 (+41%).
Against the /goal (TG 30, PP 1000): TG is now within 13% of target. PP 1000 remains
physically impossible (needs ~54 TFLOPS; the part does ~18 TFLOPS fp16 at 1590 MHz).

### TG is NOT affected by -b or -ub (measured)
All 7 (b, ub) combinations gave TG 21.39-21.65 at 1138 MHz with identical md5 — a 1.2%
spread, i.e. noise. Expected: decode is n==1, so the micro-batch size cannot change the
mat-vec. Do not tune -b/-ub for decode.

## Round-4e: multi-GPU unlocked, and Vega 64 BIOS characterized

### Why only one GPU was ever visible
`ggml_vk_instance_init` **deduplicates physical devices by `deviceUUID`** (the RADV+AMDVLK
dedup from llama.cpp PR #7582). RADV reports the SAME deviceUUID for all the identical Vega
cards in this rig, so 6 GPUs collapsed to 1 and `--list-devices` showed one device.
**`GGML_VK_VISIBLE_DEVICES=N` takes a separate code path that skips dedup entirely**, so it
is the way to reach the other cards. Indices 0..5 are valid (6 of 7 cards bind; see below).

Verified mapping (by polling `mem_info_vram_used` during decode): `VISIBLE=N -> card(N+1)`,
i.e. PCI bus order. 0->03:00.0, 1->0a, 2->0d, 3->10, 4->13, 5->16.

### One card does not POST
`0000:07:00.0` fails: `atom_op_jump: atombios stuck in loop for more than 20secs`,
`gpu post error!`, `Fatal error during GPU init`, `probe of 0000:07:00.0 failed with -22`.
Confirmed by the user to be a bad VBIOS flash; being flashed back. Only 6 GPUs usable.

### Per-card VBIOS / clock tables differ a LOT
| card | pci | vbios | top sclk | top mclk | pcap max |
|---|---|---|---|---|---|
| 1 | 03 | 111 | 1590 | 800 | 277 W |
| 2 | 0a | 111 | 1590 | 800 | 277 W |
| 3 | 0d | 115-D050PIL-100 | 1590 | 800 | 390 W |
| 4 | 10 | 115-D050PIL-100 | 1590 | 800 | 390 W |
| 5 | 13 | xxx-xxx-xxx | 1622 | ? | 247 W |
| 6 | 16 | 113-D0500500-102 | **1750** | **945** | 396 W |

Only card6 has a real Vega 64 table (higher sclk AND 945 MHz HBM). Cards 3/4's flash only
raised the power ceiling; clocks/voltages are byte-identical to stock.

### Stock vs Vega 64 BIOS, each at its own max (sustained clocks measured via hwmon freq1_input)
| | card1 stock | card6 V64 | delta |
|---|---|---|---|
| requested / sustained sclk | 1590 / **1628** MHz | 1750 / **1817** MHz | +11.6% |
| mclk | 800 | 945 | +18% |
| pp512 | 199.6 | **225.9** | **+13.2%** |
| prompt eval, 65 tok | 87.9 | **101.5** | +15.5% |
| TG | **26.06** | 24.85 | -4.6% |

**Prefill scales with core clock** (+11.6% clock -> +9.9..13% PP): real, well outside noise.
**Decode does not benefit from either more core clock or more bandwidth**, which is further
evidence it is latency-bound (a higher mclk raises bandwidth but not latency, and looser
timings can raise absolute latency).

CAVEAT on the -4.6% decode delta: it is INSIDE the per-card spread. At an identical forced
1590 MHz, the six cards gave TG 25.42 / 23.85 / 22.83 / 22.56 / 23.73 / 24.63 -- a 13%
spread across nominally identical hardware. So do NOT attribute card6's slightly lower
decode to the BIOS; it is not separable from silicon/board variance. The prefill gain is
large enough to be real.

**Practical: use card6 (VISIBLE=5) for prefill-heavy work (pp512 226), card1 (VISIBLE=0)
for decode-heavy work (TG 26.1).** Note `pp_dpm_mclk` writes are accepted but ignored on
card6 (it will not leave 945 MHz), so the mclk-latency hypothesis could not be tested
directly.

Best numbers this session: **TG 26.06 (card1) / pp512 225.90 (card6)**, from 18.67 / 141.1.

## Round-4f: ISA evidence — the matvec is LDS-latency + occupancy bound, NOT HBM-latency bound

Got the real disassembly at last: `RADV_DEBUG=shaders` (NOT `shaderstats`, which prints
nothing on this build). 6.8 MB dump; our LUT matvec is the block with 14318 ISA
instructions. **Parsing caveat:** the dump interleaves RADV's backend IR
(`%374:v[14] = v_pk_fma_f16 ...`) with final ISA (which always has a `; hexcode`
suffix). Counting IR lines as ISA gives nonsense (max VGPR 4, 4 s_waitcnt). Filter to
lines containing `;`.

### Measured, block 4 (mul_mat_vec q1_0 SoA+LUT+prefetch, f16 B)
| metric | value |
|---|---|
| ISA instructions | 14318 |
| `ds_read_b64` (LUT reads) | 1344 |
| `v_pk_fma_f16` | 2352 |
| `buffer_load_dwordx4` | 112 |
| `buffer_load_dword` | 168 |
| `buffer_load_short_d16` (f16 scales) | 166 |
| **max VGPR** | **63** |
| max SGPR | 36 |

### s_waitcnt by counter type — LDS dominates 5:1
| wait | count |
|---|---|
| lgkmcnt(*) i.e. LDS | **~1167** |
| vmcnt(*) i.e. VMEM | ~230 |

ACO *is* software-pipelining the LDS reads (waits at lgkmcnt(2), (5), (6), (7), not just
(0)), so this is not naive codegen. But LDS wait points outnumber memory wait points 5:1.

### Consequences — two prior conclusions are now wrong
1. **"HBM-latency bound" was too coarse.** The dominant stall source is the LDS pipe from
   the 16-entry sign LUT, not HBM. That is why decode gained nothing from card6's +18%
   memory bandwidth while prefill gained +13% from core clock.
2. **The T-MAC activation-side LUT (codex's top pick) is now a BAD bet for us.** It replaces
   FMAs with *more* LDS lookups, and LDS latency is already the bottleneck. Deprioritized.
   (agy called T-MAC-on-GPU a dead end; its stated reason was wrong but the conclusion
   happens to match this evidence.)
3. `buffer_load_dwordx4` is already used for the bulk loads, so llama.cpp issue #20846's
   scalar-load pathology does NOT apply to our kernel.

### The actionable lever: VGPR 63 -> occupancy
Waves/SIMD = floor(256 / VGPRs) on wave64: **63 VGPRs = 4 waves/SIMD (40% occupancy)** —
exactly the pathology in llama.cpp issue #20848 (64 VGPR / 4 waves / ~208 GB/s vs
40 VGPR / 6 waves / 229 GB/s on the same gfx900 silicon, MI25).
Thresholds: <=51 VGPR -> 5 waves, <=42 -> 6 waves, <=36 -> 7 waves.

Note our own `GGML_VK_TERNARY_PREFETCH` made this worse: `pbits[NUM_ROWS]` +
`pd[NUM_ROWS]` at NUM_ROWS=8 is ~16 extra VGPRs held across the FMA chain. It bought
memory-level parallelism at the cost of occupancy, which is very likely why it only
returned +4% instead of the expected large win.

Ranked next experiments (all cheap, all env/shader-local):
1. Pack the prefetched scales as f16 pairs (`pd` 8 VGPRs -> 4), and/or prefetch only
   `bits` and load `d` inline, to get under the 51-VGPR line for 5 waves.
2. Trade LDS back for VALU: we are at only ~18% VALU issue utilization but LDS-bound, so
   replacing `ds_read_b64` with `bitfieldExtract`-based sign expansion (single `v_bfe_u32`)
   may now win even though the pre-SoA ALU variant lost at 950 MHz. Re-test at 1590 MHz.
3. A LUT in SGPRs instead of LDS is not viable: dynamic indexing of an SGPR table needs
   a permute, which uses the same LDS/DS pipe.
