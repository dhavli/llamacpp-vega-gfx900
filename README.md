# llama.cpp Vulkan tuning for AMD Vega (gfx900 / GCN5)

Kernel work, measurements and dead ends from making large LLMs run fast on **AMD Vega 56/64**
via llama.cpp's **Vulkan** backend (Mesa RADV + ACO) — hardware with excellent memory bandwidth
and no modern matrix hardware at all: **no dp4a, no int8 dot, no matrix cores, no coopmat**, and
`v_fma_f16` at fp32 rate unless you get ACO to emit the *packed* `v_pk_fma_f16`.

Test rig: 7× Vega 56/64 (8 GB HBM2 each) on a 2-core Celeron with **3.8 GB of system RAM**,
HiveOS, every card on a mining riser at **PCIe Gen2 x1 (~0.5 GB/s)**. Everything below was
measured on that machine.

## Headline results

> **Current project direction:** production work targets sparse MoE models only, beginning
> with Qwen3.6-35B-A3B. The Bonsai Q1_0 work is retained as an experiment ledger, but its
> prefill ceiling is not usable for this deployment and it is no longer an active hosting or
> optimization target.

### Gemma 4 26B-A4B single-pod result

`unsloth/gemma-4-26B-A4B-it-qat-GGUF` at UD-Q4_K_XL (14.249 GB) runs text-only with
a full **128K q8 KV** allocation on two 8 GB Vega cards. The balanced server profile is:

```bash
GGML_VK_VISIBLE_DEVICES=0,1 GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
RADV_PERFTEST=nogttspill LLAMA_SERVER_OUTPUT_RESERVE=128 \
llama-server -m gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on --no-mmap -c 131072 -np 1 -ctk q8_0 -ctv q8_0 \
  -b 1408 -ub 384 --cache-ram 0 --ctx-checkpoints 0
```

On the 5,511-token prompt plus 128 generated tokens, two repeats reached
**721.01/24.24 and 720.76/25.35 PP/TG** (720.88/24.79 average), with identical output.
This is +21.5% PP with flat decode versus the initial b512/ub256 server control at
593.55/24.69. Ubatch 448 is pathological (105.44/2.32), while b1536/ub384 is a
prefill-only 732.47/19.75 tradeoff. q4 KV makes b2048 fit but is slower at 600.76/24.39.
At real depth, a 121,243-token fill plus 128-token generation completes at
**284.89 PP / 17.74 TG**, with the same deterministic completion and no GPU faults.
Thus 128K is operational, but shallow decode must not be extrapolated to a full context.

The shipped 252 MB MTP drafter is host-RAM-blocked on this two-card 128K profile. A diagnostic
three-card run reaches 85.96% acceptance but collapses to **378.01 PP / 5.19 TG** because
draft calls cross the Gen2 x1 links. Keep MTP off now; after the RAM upgrade, retest the exact
two-card profile before making a permanent model-level rejection.

**Qwen3.6-35B-A3B at UD-Q4_K_XL** (21.27 GiB, 256 experts / top-8, 3B active,
30 gated-delta-net + 10 full-attention layers), split across multiple cards:

| GPUs | pp512 | pp2048 | pp8192 | decode @5.6k ctx |
|---|---|---|---|---|
| **3** (`-c 8192`) | 521.1 | **641.0** | **584.9** | 30.84 |
| **4** (`-c 8192`) | 499.2 | 561.2 | 509.9 | **32.19** |
| 6 | 490.1 | 506.4 | 457.1 | 27.61 |

A sparse 35B at 4-bit beats a hand-tuned dense 27B at 1.125 bpw on **both** axes on this
hardware — 3× the prefill and better decode — because prefill is FLOP-bound and MoE simply
doesn't ask for the FLOPs these cards can't deliver. On 3 cards a real 5629-token prompt
prefills at **624 t/s** and then generates at **30.8 t/s**.

The best validated four-card top-8 profile combines the graphics queue with an explicit
layer split, ubatch 768, and per-device queue locking:
`GGML_VK_ALLOW_GRAPHICS_QUEUE=1 GGML_VK_PER_QUEUE_MUTEX=1 -ts 0.85,1.05,1.05,1.05 -ub 768`.
On the real 5,629-token prompt, repeated ub768 runs reached **583.34–584.80 PP** and
34.93–35.64 TG. Streaming PPL is 131.2037 versus 131.0328 control, and the fail-closed
`nogttspill` gate reached 585.54/34.69 with no eviction or faults. See
`docs/qwen4-optimization-roadmap.md`.

The queue-lock change removes an unnecessary process-global `vkQueueSubmit` mutex while retaining
a shared lock for wrappers that alias the same Vulkan queue. Repeated same-binary averages were
583.95 PP / 32.77 TG with the global lock and 584.71 / 34.64 with per-device locks; the final
`nogttspill` gate reached 583.74 / 35.56 with identical output. The upstream FLOP-based submission
heuristic was also tested but rejected: PP regressed 1.57% and total time improved only 1.56%.

For the production-shaped server with four resident 128k slots, keep automatic layer placement
and `-ub 512`: the shallow custom split with ub640/768 fails the no-spill fit gate. With exactly
one admitted request, adding the graphics queue raises **529.45 PP / 23.41 TG** to
**529.63 PP / 26.95 TG**, byte-identically, restoring the 500/25 per-stream SLA.
The quality-gated top-6 opt-in tier is faster still: two exact-shape runs averaged
**572.26 PP / 28.31 TG**, with identical same-tier output. Top-7 is only a mixed
551.76/25.78 tradeoff at this depth, so use top-6 for the high-throughput server tier.
For prompt-heavy workloads that can tolerate a documented quality tradeoff, top-5 repeated at
**606.57 PP / 27.99 TG**: +6.0% PP and -1.1% TG versus top-6, with identical generated text and
no GPU faults. Its PPL remains close at 132.0961 +/- 5.2769, but it deterministically exhausted
the 160-token task budget before answering the bolts task. Treat top-5 as an aggressive tier,
not a replacement for top-6. Top-4 is rejected: despite 706.43/36.52 shallow throughput, PPL
rose to 136.6585 and it invented 110 as a prime, produced the wrong sum, and violated the
array-only task. This is the measured lower-expert quality cliff; do not descend further.

Controlled clock experiments on cards 1–3 found that 900 MHz HBM raises deterministic
decode from 30.93 to **33.18 t/s** (+7.3%) with byte-identical output. 950 MHz did not improve
on that result. A 1700 MHz / 1200 mV core clock is stable and byte-identical but not useful:
decode changed 33.82 → 33.55 t/s, pp2048 578.09 → 583.70 (+1.0%), and pp8192 524.39 →
527.82 (+0.7%). Keep the stock core clock. The 900 MHz HBM gain is specific to the three-card
short probe; it is neutral on the four-card production workload and should not be deployed.

### Archived Bonsai result

**Bonsai-27B at Q1_0** reached 26.06 t/s decode and 206.4 t/s pp512 on a Vega 56, or
225.9 t/s pp512 on the Vega 64 BIOS card. The conventional packed-f16 prefill path is
roofline-limited to roughly 359 t/s even at impossible 100% peak, so it cannot meet the
serving floor. Its detailed measurements remain in `RESULTS.md`.

The best safe single-stream runtime knob found so far is
`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`: on four cards it raises decode from **29.44 to 34.87 t/s**
(+18%) with byte-identical greedy output and unchanged prefill. The gains do not add:
combining it with `RADV_PERFTEST=nogttspill` gives 32.08 t/s, or 33.78 with the async
transfer queue added. Pipeline parallelism was originally confirmed at `-b 2048 -ub 512`
and remains enabled with the new ub768 profile; its diagnostic is hidden without `--verbose`.

The tempting Q4_K prefill override
`GGML_VK_TILE_M=256,128,64,64,32,64,2,4,4,1,64` is **invalid for this model**. Despite a
19% throughput gain and roughly 140 lines of identical greedy output, its measured
perplexity is **3,583,951 vs 131.03** for the default tile. Tile correctness is kernel-path
specific: this value remains verified for Bonsai Q1_0, but must not be exported for Qwen.

For `llama-server`, export **`LLAMA_SERVER_FULL_OUTPUT_RESERVE=1`**. Prism b9599 otherwise
sets `n_outputs_max=n_parallel` (1 for a single slot), and the resulting graph reservation
collapses this hybrid model to **3.77 t/s**. Reserving the full batch, as
`llama-completion` does, restores **29.70 t/s** and raises short-prompt PP from 22.8 to 134.2,
with identical text. Disabling server context checkpoints and its prompt cache did not change
decode, so they were not the cause. The override is carried by this repo's patch.

Reducing routed experts from top-8 to top-6 is a viable experimental quality/speed option:
PPL is 131.42 vs 131.03 (a 0.30% delta, inside error), three deterministic tasks are
quality-equivalent, and the long-prompt probe improves from 519.7/29.4 to 559.1/31.1 PP/TG.
This narrow evaluation does not establish broad no-loss quality, so top-8 remains the default.

The final-runtime expert frontier adds a better-balanced top-7 tier. Top-7 reaches
**609.39 PP / 36.04 TG** with PPL 132.3556 +/- 5.2767 versus top-8's 131.2037 +/- 5.1875.
Across two fresh-process runs of eight exact/structural tasks, top-7 introduced no failure absent
in top-8 and fixed top-8's Python alias/copy error. Top-6 reaches **629.80 PP / 33.94 TG** and
matches top-7's task pass vector. Use top-7 for decode/latency, top-6 for prefill-heavy traffic,
and retain top-8 when preserving the model's trained routing semantics is more important than speed.
At four-slot 128k server shape the ranking changes: top-6 dominates both axes, while top-7 does not.
Top-5 instead favors prefill: two exact-shape runs reached 606.38/28.40 and 606.76/27.57,
with identical output. Expose it only when prompt throughput outweighs its repeatable bounded-task
regression; top-6 remains the lowest general-purpose quality-gated tier.
The final frozen Qwen server profile replaces the boolean full-output graph reservation with
`LLAMA_SERVER_OUTPUT_RESERVE=128`. That keeps top-6 at 572.17/29.54 and recovers enough VRAM for
automatic-placement ub640, which reaches **579.31 PP / 29.65 TG** with identical output. Reserve
64 regresses decode to 26.68 and ub768 still fails the fit gate.

Per-op Vulkan profiling also rules out the fused GDN kernels as a large decode lever. Across
three devices, `GATED_DELTA_NET` + `SSM_CONV_SILU` total only **1.04 ms/token**—4.0% of
instrumented GPU-op time and 2.9% of wall time. The logger requires pipeline parallelism to
be disabled in this revision, so absolute performance is perturbed, but even a perfect 2×
kernel improvement would move wall time by only ~1.5%.

The Qwen MoE fusion matcher does fire: perf logs contain
`TOPK_MOE_EARLY_SOFTMAX_NORM`, and disabling all Vulkan fusion drops decode from 29.44 to
24.87 t/s (−15.5%) with byte-identical text. The suspected view-node matcher miss is not
present in this build; the existing fusion path should be preserved.

Note the direction: MoE prefill gets **worse** with more cards (641 → 561 → 506 at pp2048),
the opposite of the dense model, which *gained* from 3 cards at long prompts. A3B does roughly
a ninth of the matmul per token, so per-stage work is small relative to the fixed per-boundary
copy — sparsity cuts compute but not the number of layer boundaries. Decode peaks at **4**
cards, not 3: at 3 cards VRAM sits at 7.0–7.2 GiB of 7.98 per card, and the slack from a fourth
card outweighs the extra boundary. Past that, boundary cost dominates.

### Long context and concurrent slots

Only 10 of the 40 layers hold a KV cache (2 KV heads × 256 head_dim), so KV costs just
**20 KB/token** — 2.62 GB at 128k, or 1.31 GB with `-ctk q8_0 -ctv q8_0`. That makes this model
unusually cheap to serve at depth on 8 GB cards. Single slot, 6 cards, `-c 131072`:

| KV type | prefill | decode |
|---|---|---|
| f16 | 418.1 | 8.99 |
| **q8_0** | 475.5 | **21.00** |

q8_0 KV is **mandatory** at long context, not an optimisation. A 2.3× decode difference cannot
come from the bandwidth of an extra 1.3 GB (~6 ms/token at most against the 63 ms observed), so
f16 at 128k is tipping a card into spill — the 10 attention layers don't divide evenly across
6 devices, so one card takes several of them.

Concurrent slots, all 6 cards, q8_0 KV, `-npp 8192 -ntg 128`:

| slots | total ctx | prefill t/s | aggregate decode t/s | combined t/s |
|---|---|---|---|---|
| 2 | 262,144 | 437.3 | 33.92 | 369.6 |
| 4 | 524,288 | 446.8 | 46.77 | 394.9 |
| 8 | **1,048,576** | 445.2 | **56.10** | **402.2** |

**8 slots × 128k allocates and runs on 6 × 8 GB.** Aggregate decode scales with slot count
because batching amortises the per-boundary copy over more sequences, while prefill saturates
at ~445 t/s regardless — so prefill is the binding constraint for serving, not decode. Caveat:
KV is *allocated* for 128k/slot but only 8320 tokens/slot were populated here, so these decode
figures are at shallow depth; expect roughly 25% less at genuinely full slots, extrapolating
from the single-slot 27.6 @16k → 21.0 @128k measurement.

## What's in the patch

`patches/0001-vulkan-wide-ternary-mmv-gcn-knobs.patch` — one squashed commit against
PrismML llama.cpp `b9596-9fcaed7`. Every change is behind an env var so it can be A/B'd.

- **`PACKED_K` mul_mm** (`GGML_VK_MM_PACKED=1`) — pairs even/odd k into `f16vec2` lanes so ACO
  emits `v_pk_fma_f16` instead of two scalar `v_fma_f16`. GCN runs scalar fp16 at fp32 rate;
  only the packed form is 2×. **+33% prefill.**
- **LDS sign-nibble LUT matvec** (`GGML_VK_TERNARY_LUT=1`) — maps 4 sign bits to `f16vec4(±1)`
  through a 16-entry shared table: one `ds_read_b64` + 2 packed FMAs per 8 weights, and the ±1
  encoding removes the `sum(b)` correction term entirely. **+24% decode.**
- **Wide n==1 matvec + row/workgroup knobs** (`GGML_VK_TERNARY_ROWS`, `_BS`,
  `GGML_VK_GCN_SUBGROUP_REDUCE`) — 8 rows per workgroup at wg=64 is the optimum here.
- **SoA weight repack at load time** (`GGML_VK_TERNARY_SOA=1`) — one coalesced dword per 32
  weights. Correct and required by the LUT path; neutral on its own.
- **Software-pipelined matvec prefetch** (`GGML_VK_TERNARY_PREFETCH=1`) — hoists all per-row
  weight loads ahead of the FMA chains. **+5.7%.**
- **Runtime warptile override** (`GGML_VK_TILE_{L,M,S}` = 11 comma-separated uints
  `{threads,BM,BN,BK,WM,WN,WMITER,TM,TN,TK,subgroup}`) — makes mul_mm tiles tunable without
  a rebuild. Note `mul_mat_l` is disabled for Q1_0 by the shmem check, so the **medium** tile
  is the one that runs.
- **`SIGMOID`+`MUL` graph fusion** with a scan-based matcher that skips view-only nodes.
- **Snapshot-free gated-delta-net rows mode** — `ggml_gated_delta_net_rows` with K=1, decoupled
  from the rollback ring, so state can be read from cache rows without writing per-token
  snapshots.

Build with the flake (pinned nixpkgs + prism source, patches applied automatically):

```
nix build .#vega-runtime          # llama.cpp + Mesa RADV closure
nix copy --to ssh://root@RIG --no-check-sigs ./result-runtime
```

Production env for Bonsai-27B Q1_0:

```
GGML_VK_TERNARY_ROWS=8 GGML_VK_GCN_SUBGROUP_REDUCE=1 GGML_VK_TERNARY_SOA=1 \
GGML_VK_TERNARY_LUT=1 GGML_VK_TERNARY_PREFETCH=1 GGML_VK_MM_PACKED=1 \
GGML_VK_TILE_M=256,128,64,64,32,64,2,4,4,1,64
```

## Multi-GPU on Vulkan: what actually happens

`--split-mode row` and `-sm tensor` are not usable here — `ggml_backend_split_buffer_type` is
only exported by the CUDA backend, so **layer split is the only mode**. Layer split is pipeline
parallelism, and it behaves very differently for prefill and decode:

| Bonsai-27B Q1_0 (dense) | pp512 | pp2048 | pp8192 | decode |
|---|---|---|---|---|
| 1 GPU | 206.4 | 200.5 | 143.4 | 25.65 |
| 3 GPUs | 187.9 | 238.0 | **210.1** | 2.32 † |
| 6 GPUs | 176.8 | 185.9 | 169.0 | 17.56 |

**Prefill gets faster with more cards, and the gain grows with prompt length** (−9% at pp512,
+19% at pp2048, +47% at pp8192) because llama.cpp submits ubatches asynchronously: ubatch *N+1*
can start on GPU 0's layers while ubatch *N* is still finishing on GPU 2. At pp512 there's only
one ubatch, so there's nothing to overlap and you just pay the boundary cost.

**Decode has nothing to overlap**, so every layer boundary is a blocking cross-device copy —
`ggml_backend_vk_cpy_tensor_async` refuses cross-device transfers and falls back to host-staged
`ggml_vk_buffer_copy` with a fence wait. There is no P2P and no `VK_KHR_device_group`. Measured
cost at 6 GPUs: ~18 ms/token over 5 boundaries, i.e. **~3.6 ms per boundary**, which is roughly
half the token time. Fewer cards is better for decode, as long as the model fits.

† the 2.32 t/s figure is under investigation — it is inconsistent with the per-boundary cost
implied by the 6-GPU point, and was measured while a large download was saturating the host.

**Watch out for GTT spill, and don't trust free-VRAM readings to detect it.** With `-ngl 99`
set explicitly, llama.cpp's auto-fitter gives up (`failed to fit params to free device memory:
n_gpu_layers already set by user to 99, abort`) and allocation proceeds anyway; the kernel then
transparently evicts buffers to GTT and every access crosses PCIe. Symptom: prefill collapsing
from 585 t/s to 9.3 t/s on the same cards, with decode at 1.5 t/s.

The trap is that **an evicted buffer no longer counts as VRAM used**, so `mem_info_vram_used`
reads *healthier* the worse the spill is. The 3-card config that works (`-c 8192`) sits at
7041/7034/7219 MiB of 7.98 GiB per card; the 4-card config that spilled reported only
5.7–5.9 GiB per card and looked like it had 2 GB spare. Judge fit by whether throughput matches
`llama-bench` on the same devices, not by reported free memory. Also note `token_embd` and
`output.weight` (~540 MB each at Q8_0 with a 248320-token vocab) both land on the main device,
so the split is less even than layer counts suggest.

**Don't trust sysfs/lspci link speed on Vega — read the bridge, not the GPU.** Vega 10 puts a
two-level PCIe bridge on the package, and both the GPU function's `lspci -vv LnkSta` and
amdgpu's `current_link_speed`/`current_link_width` report the *on-die* hop — a very healthy
"8 GT/s x16" — regardless of how the card is actually connected. The real host-facing link is
the outermost bridge (`01:00.0`-style) to its root port, and `pp_dpm_pcie` tells the truth: on
this rig every card offers exactly one state, `5.0GT/s, x1` — Gen2 x1 risers, ~0.5 GB/s each
way, even for the card in the CPU's x16 slot. That's fine for layer-split *decode* (the hidden
state is ~4 KB/token/boundary; latency dominates, not bandwidth) but each prefill ubatch moves
megabytes per boundary through host RAM, which is part of why MoE prefill decays monotonically
with card count.

**Serving topology: fewer cards per instance wins, and host RAM is the real wall.** At an
identical serving load (4 slots × 128k, q8_0 KV, 8k prompts) 4 cards beat 6 cards on both axes:
prefill 471 vs 425 t/s, aggregate decode 52.2 vs 48.6 — two fewer layer boundaries is worth
more than two extra GPUs. But capacity is card-bound (8×128k needs ~32.3 GB and does *not* fit
4×7.98 GB), and running **two instances on disjoint cards** — the only way to scale aggregate
prefill, since slots share one compute budget — is blocked by host RAM on a 3.8 GB box: every
attempt was OOM-killed, **and zram does not help**. The killed processes held ~150 KB of anon
RSS; the pressure is pinned GTT pages (Vulkan host-visible staging), which swap cannot touch.
Budget real RAM for `n_instances × ~2.75 GB` plus page cache, or benchmarks spend ~80% of
wall-clock re-reading the model from disk.

The matvec is a well-established local optimum. All of these were built and measured, at
1590 MHz, and lost:

- **VGPR diet / higher occupancy.** Dumped ISA: the non-prefetch variant is 47 VGPR / 5 waves
  and gives 24.57 t/s; the prefetch variant is 63 VGPR / 4 waves and gives 26.01. The
  *lower*-occupancy variant is faster. What tracks performance is `lgkmcnt` waits per unit
  work, not waves/SIMD.
- **LDS → ALU** (pure ALU sign expansion): 21.4 vs 26.0, i.e. 20% slower, despite the kernel
  being LDS-stall bound at only ~18% VALU utilisation.
- **Byte-indexed LUT** (256 × 16 B = 4 KB LDS, 8 weights per lookup): 22.7 vs 25.9. 11% slower
  from **LDS bank conflicts** — a 16-byte entry spans 4 of 32 banks, so entries 8 apart
  collide. This is exactly why the nibble LUT wins: 16 × 8 B = 128 B maps 1:1 onto bank pairs
  and is the largest conflict-free table.
- **fp16 staging in the matvec** (u32→f16 conversion overhead): −9%.
- **LDS double-buffering in mul_mm** at doubled footprint: 133 vs 138 — halves workgroup
  occupancy per CU, costing more than the saved barrier drains.
- **`LOAD_VEC_A=16`** in mul_mm: −5%.
- **ROCm/HIP on gfx900**: decode 12.4 vs 15.1, prefill 35.0 vs 98.9 against Vulkan on identical
  hardware. gfx900 has no dp4a, and TheRock excludes `composable_kernel`/`rocWMMA` for it.
- **`-b` (logical batch)**: a pure no-op for prefill. `-ub` is the real knob.
- **Pre-formatted sign masks** (sign bits pre-placed at bit 15/31 of a dword so a `v_xor_b32`
  makes ±1 with no LDS): a 32-bit mask covering 2 weights is 16 bits of storage *per weight* —
  a 16× expansion that destroys the whole point of 1-bit weights.
- **"Fast" warptiles that turned out to compute garbage.** Sweeping `GGML_VK_TILE_M` on the
  Q4_K MoE path produced pp2048 of 996 (BK=32), 1128 (BN=256) and **1526** (TM=8) against the
  GCN default's 506 — up to 3×. All three are **wrong**: under greedy decode with an identical
  prompt, the default emits coherent text and TM=8 emits a run of `/` characters. They are fast
  because they skip work.

  This is a trap worth internalising: `llama-bench` never inspects output, and llama.cpp's
  `ggml_vk_matmul_shmem_support` only validates that the tile *fits in LDS* — it does not check
  that `BM`/`WM`/`WMITER`/`TM`/subgroup are geometrically consistent. An invalid tile therefore
  loads, runs, and benchmarks beautifully. **Every warptile candidate must be output-diffed
  against the default** (`llama-completion --temp 0 --ignore-eos`, byte-compare the
  continuation) before its throughput means anything.

## Measurement lessons

- **Rank by critical path, not by per-op totals.** Removing 48 per-layer state gathers cut
  `GET_ROWS` from 3.31 ms to 0.86 ms of GPU op time and moved wall clock by 0.37 ms — the gather
  reads the KV cache and is independent of the current layer's activations, so it already
  overlapped with matvec work.
- **Decode is 100% GPU-bound**, not submission-bound: over 10 s of steady single-GPU decode the
  process accrued 0 ms user + 0 ms system time (command buffers are reused).
- **Read `pp_od_clk_voltage` before forcing a DPM state.** A mining profile had pinned every
  state to a flat 800 mV, which made 1269 MHz and up hang and looked exactly like a degraded
  card. At stock voltages every state up to 1590 MHz was stable. This one cost hours and
  invalidated three rounds of roofline math done against the wrong clock.
- **`RADV_DEBUG=shaders` interleaves backend IR with final ISA.** Only ISA lines carry a
  `; hexcode` suffix; counting IR lines as ISA yields nonsense (max VGPR 4).
- **`GGML_VK_VISIBLE_DEVICES` is the only way to see more than one identical GPU.**
  `ggml_vk_instance_init` deduplicates by `deviceUUID`, and RADV reports the same UUID for every
  identical card, so 6 GPUs collapse to 1.
- Don't poll `gpu_busy_percent` / `temp1_input` / `power1_average` under load — it floods dmesg
  with SMU read errors that mask real faults.
- A GPU hang here is **not** recoverable without a reboot: it leaves unkillable D-state
  processes and wedged `kworker/*+ttm` threads, after which every later benchmark silently
  thrashes.

## Repo layout

| path | what |
|---|---|
| `patches/0001-*.patch` | the kernel work, one squashed commit |
| `flake.nix` / `flake.lock` | pinned reproducible build (llama.cpp + Mesa RADV closure) |
| `RESULTS.md` | full chronological results with methodology per round |
| `ANALYSIS.md` | up-front architecture analysis and predictions (some later falsified) |
| `ROUND3-NOTES.md` | working notes, including every dead end in detail |
| `benchmarks/llm/index.jsonl` | machine-readable ledger, one record per measurement |

## Licence

The patch is against llama.cpp and inherits its MIT licence. Everything else here is MIT.
