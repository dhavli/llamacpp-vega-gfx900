# Round-3 work notes (toward TG 30 / PP "bigger the better")

State after round 2 (commit 1e6b91f, runtime store `pk52zfmn...`): TG 18.65, pp512 141.1.
Decode budget 53.6 ms/token = ~33 ms matvec + ~15-20 ms small-op dispatch swarm.

## (a) TG: fuse delta-net elementwise chains (est. −8-12 ms/token)
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

## (b) PP: mul_mm LDS double-buffering (est. +15-25%)
mul_mm.comp block loop: load_a/load_b → barrier → math → barrier, ~160 iterations of
full pipeline drain. Double-buffer buf_a/buf_b (2× shmem: 2×16.9 KB at BM128/BN64 fits) and
prefetch tile k+1 during math on tile k. pk-issue efficiency currently 41%.

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
