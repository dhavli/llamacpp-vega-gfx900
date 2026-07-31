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
