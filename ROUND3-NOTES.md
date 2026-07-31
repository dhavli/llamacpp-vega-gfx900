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
