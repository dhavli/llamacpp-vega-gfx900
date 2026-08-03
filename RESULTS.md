# Bonsai-27B on Vega 56 — Measured Baseline (2026-07-30)

**It works.** A 27B vision-language model with 128k context, running on a single 2017
8 GB Vega 56 via Vulkan, at ~13.4 tok/s. No code was written — prebuilt binary, stock RADV.

Full ledger: `benchmarks/llm/index.jsonl` (12 records, including invalid/artifact runs).

---

## 1. Environment

| | |
|---|---|
| Host | `RIG4AMDRX580Vega`, HiveOS / Ubuntu 22.04.5, kernel 6.6.0-hiveos |
| GPUs | **7× AMD Radeon RX Vega 56** (gfx900, 8176 MiB each), all PCIe **8GT/s x16** |
| CPU / RAM | **2 cores** (SSE4.2 tier, no AVX2), **3810 MiB RAM**, no swap initially |
| Driver | Mesa **23.2.1** RADV (installed `mesa-vulkan-drivers`, `libvulkan1`, `vulkan-tools`) |
| Runtime | PrismML llama.cpp prebuilt **b9596-9fcaed763**, `bin-ubuntu-vulkan-x64` — no compiling |
| Model | `Bonsai-27B-Q1_0.gguf` (3.53 GiB, 1.125 bpw) and `Ternary-Bonsai-27B-Q2_0.gguf` (6.66 GiB) |

Cards are **Vega 56, not 64** — 410 GB/s, so ~15% below the Vega 64 figures in `ANALYSIS.md`.

ggml's own device probe confirmed every hardware prediction from the analysis:

```
AMD Radeon RX Vega 56 (RADV VEGA10) (radv) | uma: 0 | fp16: 1 | bf16: 0 |
warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: none
```

`int dot: 0` and `matrix cores: none` are the empirical confirmation of no dp4a and no
coopmat on gfx900. `vulkaninfo` independently reported
`integerDotProduct4x8BitPackedSignedAccelerated = false` and zero `cooperative_matrix`
occurrences.

---

## 2. Headline numbers — single Vega 56, Q1_0, q4_0 KV, flash attention on

| Context | Decode (tg) | ms/token | Peak VRAM | Headroom |
|---|---|---|---|---|
| 4 096 | **13.14 t/s** (199 tokens) | 76.1 | — | — |
| 32 768 | **13.65 t/s** | 73.2 | 4 709 MiB | 3 467 MiB |
| **131 072** | **13.43 t/s** | 74.5 | **6 414 MiB** | **1 762 MiB** |

Prefill: **pp512 = 92.3 ± 0.1 t/s** (Q1_0), **83.4 ± 2.0 t/s** (Q2_0).

Output is fully coherent — correct, well-structured technical prose with markdown
structure. This is not a "it emits tokens" result; the model is working properly.

### The standout finding: decode is context-independent

13.14 t/s at 4k → 13.65 at 32k → 13.43 at 128k. **A 32× increase in context costs
essentially nothing in decode speed.** That is the hybrid ~75%-linear-attention backbone
delivering exactly what it promises. A conventional transformer would degrade badly by 128k.

For this rig it means the 8 GB VRAM limit — not attention cost — is the binding constraint,
and 128k already fits.

---

## 3. Two measurement traps we hit (both mine, both instructive)

### 3.1 `--no-mmap` is mandatory here — 21× difference

With mmap on (the default), decode ran at **0.54 t/s**. With `--mmap 0`: **11.34 t/s**.

Measured cause, during pure decode with weights already resident in VRAM:

```
majflt/s=432  diskread=92MiB/s  swap_used=25MiB  cache=2673MiB  vram=6509MiB
majflt/s=310  diskread=97MiB/s  swap_used=25MiB  cache=2679MiB  vram=6509MiB
```

~400 major faults/sec and **~93 MiB/s sustained disk read while generating** — the page
cache can hold only ~2.6 GB of a 6.66 GiB mapping on a 3810 MiB host, so weights are
re-read from disk continuously. The SATA disk, not the GPU, was the bottleneck (~172
MiB/token).

`--no-mmap` does **not** increase host RAM use with full GPU offload — llama.cpp streams
tensor-by-tensor into VRAM. Host stayed flat at ~2580 MiB available. This vindicates the
`rdna2-gpu-constraints` skill's "always `--no-mmap`" rule; I initially argued against it on
low-RAM grounds and was wrong.

### 3.2 The 0.54 t/s "cliff" is a `llama-bench` artifact, not a backend limit

`llama-bench` sets `n_ctx = n_gen`. Sweeping generation length:

| n | 16 | 32 | 48 | 64 | 80 | 96 | **104** | 112 | 120 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|
| t/s | 13.77 | 13.00 | 13.09 | 13.64 | 13.19 | 13.32 | **0.54** | 0.54 | 0.54 | 0.54 |

Flat to 96, then a uniform 25× per-token penalty (1.85 s/token) once generation nearly
fills the context. Mechanism, measured in the slow regime:

```
gpu_busy=100%  cpu_idle=93.9%  diskread=0MiB/s
gpu_busy=99%   cpu_idle=90.9%  diskread=0MiB/s
```

GPU pegged, CPU idle, no I/O → a **GPU-side work explosion in a Vulkan shader** when the
context is full. Not CPU fallback (my first hypothesis) and not I/O (my second).

**It does not reproduce with a real context**: `-c 4096` generating 199 tokens sustains
13.14 t/s with `graphs reused = 198`. So it is a harness pathology, harmless in production
— but worth reporting upstream, and a reason to **never benchmark this architecture with
`llama-bench` tg tests**. Use `llama-completion`/`llama-server` with an explicit `-c`.

---

## 4. How much headroom is left

Decode at 13.43 t/s × 3.53 GiB = **47.4 GiB/s ≈ 51 GB/s effective, ~12% of the Vega 56's
410 GB/s.** For reference, Llama-2-7B Q4_0 on Vega 56 achieves ~50 t/s × 3.56 GiB ≈ 191
GB/s (**~47%**).

So we are running at roughly **a quarter of the memory efficiency a mature quant achieves
on this same silicon.** That is not a hardware wall — it is unoptimised kernels, and it
means a 2–3× decode improvement is plausibly available.

### Prime suspect, identified in source

Q2_0/Q1_0 are the only quant types in the Vulkan backend with **no `_packed16`/`_packed32`
struct variants**. Compare the hot 4-element dequant path:

```glsl
// Q4_0 — 16-bit aligned load via the packed alias, 4 values:
const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);

// Q2_0 — single BYTE load off the unpacked struct, 4 values:
const uint byte_val = uint(data_a[a_offset + ib].qs[iqs / 4u]);
```

`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`, `iq1_s`, `iq1_m` all define
`A_TYPE_PACKED16`; `Q1_0`/`Q2_0` do not. A 128-weight ternary block therefore issues 32
byte-granularity loads where a single dword load could fetch 16 weights. On GCN, byte
access through an SSBO is materially worse than dword access.

**This is the first optimisation to attempt**: add `block_q2_0_packed16`/`packed32` (and
Q1_0 equivalents) plus packed load paths in `dequant_funcs.glsl`. It is self-contained,
low-risk, and upstreamable — it would benefit every GCN/Vulkan user of these formats, not
just us.

---

## 5. Reproducing

```bash
# one-time host setup
apt-get install -y mesa-vulkan-drivers libvulkan1 vulkan-tools libgomp1 libcurl4

B=/root/bonsai/llama-prism-b9596-9fcaed7
export LD_LIBRARY_PATH=$B GGML_VK_VISIBLE_DEVICES=0

$B/llama-completion -m /root/models/Bonsai-27B-Q1_0.gguf \
  --device Vulkan0 -ngl 99 -c 131072 -fa on -ctk q4_0 -ctv q4_0 \
  --no-mmap -no-cnv -p "..." -n 200 --no-warmup
```

Non-obvious flags: `GGML_VK_VISIBLE_DEVICES` / `--device` are needed to exclude `llvmpipe`
(RADV enumerates it as a device and it would be catastrophically slow); `--no-mmap` per
§3.1; `-ctk/-ctv q4_0` for the 4-bit KV cache that makes 128k fit.

Gotchas: `llama-cli` rejects `-no-cnv` ("use llama-completion instead"); `llama-bench` wants
`--mmap 0` while `llama-completion` wants `--no-mmap`.

---

## 5b. Optimization round 1 (2026-07-30 afternoon) — +15% decode, reproducible builds

Everything now builds from `flake.nix` (pinned nixpkgs + prism `b9596`, patches applied
from `patches/`), and deploys to the rig as a nix closure with its own Mesa — the rig's
2023 driver stack is no longer a variable.

| Config | Decode | pp512 |
|---|---|---|
| Original prebuilt + Mesa 23.2.1 | 13.14 t/s | 92.3 |
| Same code, **Mesa 26.1.5** (nix closure) | 13.81 t/s | 98.9 |
| + `patches/0001` wide ternary matvec, `GGML_VK_TERNARY_ROWS=8` | **14.90–15.08 t/s** | 98.9 |

- The patch: dedicated n==1 `mul_mat_vec` variant for Q1_0/Q2_0 — 32 weights/thread/iter
  via packed16 loads, activations staged in registers across rows, dot computed as
  `fma(code, b) − sum(b)` without materializing weights. Generic path kept for batch 2–8
  (the wide variant's register pressure regresses it). Greedy outputs identical to baseline.
- Knobs added: `GGML_VK_TERNARY_ROWS` (8 optimal; 16 worse — VGPR limit),
  `GGML_VK_GCN_SUBGROUP_REDUCE` (≈1% — GCN's shmem reduction wasn't the bottleneck).
- Per-op profile after patch: big matvec 186→155 µs, still only ~81 GB/s of 410 —
  **latency/occupancy-bound, not instruction-bound**. Next 2× needs fp16 packed math or a
  fundamentally different tiling, plus the ~14 ms/token of small-op dispatch overhead.
- `llama-bench` tg is still unusable on this arch; decode numbers above are
  `llama-completion` with real contexts, `--temp 0`, fixed 128 tokens.

## 5c. ROCm/HIP baseline — works on gfx900, loses to Vulkan

nixpkgs ROCm **7.2.3 still ships gfx900 targets**; the HIP closure built first try and
**all 7 Vega 56 enumerate as ROCm devices** on the HiveOS 6.6 kernel. Bring-up fixes, in
order: (1) nixpkgs `libLLVM` assumes AVX → SIGILL on the rig's Celeron; worked around by
interposing AMD's official generic-x86 comgr (`/root/bonsai/stub`, with nix zlib/zstd
symlinks — the HIP runtime resolves comgr purely via dlsym, zero named imports).
(2) A truncated `libdrm_amdgpu.so` from an interrupted rsync — the failure mode that
motivated installing Nix on the rig (see §5d). One watchdog reboot occurred during
bring-up with the corrupt lib; the full matrix ran stable afterwards.

Measured, single Vega 56, Q1_0:

| | tg128 | pp512 |
|---|---|---|
| ROCm (fa, q4 KV) | 12.0 t/s | **35.0** |
| Vulkan patched (same feats) | **14.9–15.1 t/s** | **98.9** |

**Verdict: Vulkan wins decisively on gfx900** — decode +25%, prefill 2.8×. Without dp4a
the HIP quant paths can't compete, exactly as `ANALYSIS.md` §6.4 predicted. ROCm stays a
recorded baseline, not a direction.

## 5d. Deployment: Nix installed on the rig

Single-user Nix 2.35.1 now runs on the rig (nixbld group created; binaries symlinked to
`/usr/local/bin`; `nix-command flakes` enabled). Deploys are now
`nix copy --to ssh://root@RIG-HOST --no-check-sigs ./result-runtime` — hash-verified
and registered, no more silent truncation. The earlier rsync-shipped paths were
re-registered by the first `nix copy`.

## 6. Next steps, in priority order

1. **Multi-GPU scaling.** 7 idle cards on x16 links. Layer-split 2–4 cards for Q2_0 (better
   quality) at 128k, and measure whether x16 changes the row-split calculus — my analysis
   assumed x1 risers and advised against row-split; on x16 Gen3 it deserves a real test.
2. **Throughput mode.** One card holds Q1_0 at 128k, so run **7 independent instances** for
   ~94 tok/s aggregate. Likely the best use of this rig.
3. **The packed-load optimisation** (§4). Requires a source build — note 2 cores means a
   long compile; consider building elsewhere and copying binaries.
4. **Q2_0 vs Q1_0 quality.** Q2_0 costs ~2× VRAM for unknown quality gain. The repo ships
   `.eval_results/` (aime_2026, gsm8k, mmmu_pro) — compare before committing.
5. **Speculative decoding.** `Bonsai-27B-dspark-Q4_1.gguf` (1.79 GB) claims 1.34× on CUDA.
   Fits alongside Q1_0 on one card.
6. **ROCm experiment** (low priority, per `ANALYSIS.md` §6.4). Note this host has
   `hive-rocm-5.6` but llama.cpp needs HIP ≥ 6.1. Useful reference found:
   `jcbtc/Ternary-Bonsai-27B-AMD-Strix-Halo` ships a `build-from-source.sh` for a ROCm
   Bonsai build (gfx1151, but the scaffolding is instructive).
7. **Report the `llama-bench` artifact** (§3.2) upstream.

## 7. Rig state left behind

- Installed: `mesa-vulkan-drivers`, `libvulkan1`, `vulkan-tools`, `libgomp1`, `libcurl4`, `curl`
- Added `/swapfile.bonsai` (8 GB, `vm.swappiness=10`) as OOM insurance — **not needed in the
  end** (peak swap use 25 MiB); remove with `swapoff /swapfile.bonsai && rm /swapfile.bonsai`
- `/root/models/` (10.9 GB of GGUFs), `/root/bonsai/` (binaries, logs, `bench-v1.sh`, `vram.sh`)
- No miners were running before or after; no HiveOS services altered; all 7 GPUs idle at 7 MiB

## 5e. Optimization round 2 (2026-07-30/31 evening) — decode +42%, prefill +53%

Goal set: PP 1000 / TG 30 aspirational. Final state this round (patch v8, one env set):

| metric | baseline | round 1 | **round 2** | config |
|---|---|---|---|---|
| decode | 13.14 t/s | 15.08 | **18.65 t/s** | `TERNARY_ROWS=8 GCN_SUBGROUP_REDUCE=1 TERNARY_SOA=1 TERNARY_LUT=1` |
| pp512 | 92.3 t/s | 99.2 | **141.1** (135.1 with SoA on) | `MM_PACKED=1 TILE_M=256,128,64,32,32,64,2,4,4,1,64` |

**What moved the needle:**
1. **PACKED_K mul_mm** (`GGML_VK_MM_PACKED=1`): GCN runs *scalar* `v_fma_f16` at fp32 rate —
   only `v_pk_fma_f16` is double-rate. Pairing even/odd k in the two lanes of one `f16vec2`
   accumulator per output halves the FMA instruction count: pp512 98→131 (+33%).
2. **Warptile override** (`GGML_VK_TILE_M`, runtime env, no rebuild): `mul_mat_l` is disabled
   for q1_0 by the shmem check, so the *medium* tile is what actually runs. Scan found
   128×64, BK=32, 4×wave64, TM=TN=4, WM=32/WN=64 → 141.1. (BK 16/64 −20%, bigger tiles worse.)
3. **LDS lookup-table matvec** (`GGML_VK_TERNARY_LUT=1`): sign-bit nibble → `f16vec4(±1)` via a
   16-entry shared table; one `ds_read_b64` + 2 packed FMAs per 8 weights replaces the
   bfe+cvt+fma chains and eliminates the `sum(b)` correction. Decode 66.4 → 53.6 ms/token
   (+24%), output **bit-identical**. Requires SoA (below).
4. **SoA weight repack** (`GGML_VK_TERNARY_SOA=1`): Q1_0 mul_mat weights repacked at load into
   [64×f16 d][64×16B qs] groups; qs becomes one lane-coalesced dword per 32 weights.
   Perf-neutral alone (falsified the coalescing theory) but is the substrate for the LUT
   kernel's single-dword bit loads. Gotcha: weights stream through
   `ggml_backend_vk_set_tensor_2d_async` in chunks — repack hooks that path with a host shadow.

**Falsified (measured, don't redo):** fp16 packing alone (cvt overhead, −9%); occupancy
(7 waves vs 4 — no change); load coalescing alone (SoA neutral); LOAD_VEC_A=16 load_a (−5%,
LDS write conflicts); BK_STEP=4 ds_read_b64 (neutral — ACO already merges); workgroup 128/256
for matvec (−9/−19%); rows ≠ 8 for the wide path. The Q1_0 matvec was **ALU-bound**
(~3 VALU/weight), not memory-bound: skip-A probe deltas overstated load cost because constant
bits let the compiler fold the mask ALU.

**Where the remaining time goes (decode, 53.6 ms):** ~33 ms matvec (floors: 9.2 ms DRAM,
~3 ms pk-issue) + ~15-20 ms across ~450 tiny dispatches/token (GET_ROWS ×97, RMS_NORM ×129,
SCALE/CPY/MUL/ADD/L2_NORM ~96 each — delta-net state plumbing, 10-35 µs flat each).
Prefill: mul_mm at ~8.6 TFLOPS effective = 41% pk-issue efficiency; rest is load_a dequant +
per-BK-tile barrier drains.

**Round-3 roadmap (est. gains):** (a) fuse delta-net elementwise chains (SIGMOID+MUL gate,
L2_NORM pairs, out-ids GET_ROWS elision at n==1) → decode −8-12 ms; (b) mul_mm LDS
double-buffering to overlap tile loads with math → pp +15-25%; (c) generic n=2..8 dmmv is
6.5× the n=1 cost per token (SoA generic path slow) — worth a look for short-prompt latency.
Physics honesty: PP 1000 needs ~54 TFLOPS effective; the card's fp16 peak is 21 → ceiling
~390, realistic ~250-300. TG 30 = 33 ms/token — reachable only with both (a) and more matvec.

## 5f. Optimization rounds 3-4 (2026-07-31) — decode 26.1 t/s, prefill 226 t/s

Supersedes the round-3 roadmap at the end of §5e; several of its assumptions were wrong.

### Headline

| | round-2 end | now | vs original baseline |
|---|---|---|---|
| decode (TG) | 18.67 | **26.06** t/s | 13.14 → +98% |
| prefill pp512 | 141.1 | **225.9** t/s | 92.3 → +145% |
| pp8192 | — | 156.4 (`-ub 1024`) | — |

Best decode is card1 (stock BIOS, 1590 MHz); best prefill is card6 (Vega 64 BIOS,
1750/945). Config: the §5e env plus `GGML_VK_TERNARY_PREFETCH=1` and
`GGML_VK_TILE_M=256,128,64,64,32,64,2,4,4,1,64`, plus `-ub 1024` for long prompts.

### The single biggest win had nothing to do with kernels
**The GPUs were pinned to 950 MHz by a HiveOS mining profile that flattened every DPM
state to 800 mV** — not the 1474 MHz stock boost every roofline in §4/§5e assumed. Raising
the clock is worth +34% decode and +67% prefill on its own.

A cautionary sequence: forcing higher DPM states while that 800 mV table was active hung
the GPU (`amdgpu_job_timedout: ring comp_1.1.0 timeout`), which left unkillable D-state
processes and wedged TTM kworkers, requiring a reboot — and silently poisoned every
benchmark until then. I concluded from three such hangs that 1138 MHz was a hard ceiling
and the card was degraded. **That was wrong.** Once the OC was cleared the real per-state
voltages appeared (1269 needs 1000 mV, 1590 needs 1200 mV); my "fixes" at 900/950 mV were
still under spec. At stock voltages the whole ladder to 1590 MHz is stable at 41 °C.
**Always read `pp_od_clk_voltage` before forcing a DPM state.**

### Corrected physics (§5e's numbers used the wrong clock)
At 1590 MHz the fp16 packed peak is ~18 TFLOPS, not 21. PP 1000 would need ~54 TFLOPS
effective — **physically impossible**. Realistic prefill ceiling is ~250-280.
Decode: ~26 t/s is close to this kernel's practical limit (see refutations below).

### Batch knobs: `-b` does nothing, `-ub` is prefill-only
`-b` swept 512..8192 changes throughput by 0.1% (noise) — it only controls how many tokens
reach `llama_decode` before being split into `-ub` chunks, so it never changes GEMM shapes.
`-ub` matters only on long prompts and is **non-monotonic**: pp8192 = 119.8 (ub 512),
134.7 (1024), 131.2 (1536), 137.9 (1792). `ub=2048` OOMs because the logits buffer is
`n_vocab × ub × 4` and this model's vocab is 248320 → exactly 2.03 GB. Neither knob affects
decode at all (7 combinations, 1.2% spread) since decode is n==1.

### Multi-GPU was invisible, not absent
`ggml_vk_instance_init` deduplicates physical devices by `deviceUUID`, and RADV reports the
same UUID for every identical Vega in the rig, so 6 GPUs collapsed to one.
**`GGML_VK_VISIBLE_DEVICES=N` skips that path** and reaches all of them (`N` → `card N+1`).

### What the matvec is actually bound by, and what does NOT help
ISA via `RADV_DEBUG=shaders` (`shaderstats` is silent on this build): `s_waitcnt lgkmcnt`
(LDS) outnumbers `vmcnt` (VMEM) ~5:1, so the kernel is **LDS-stall bound**, not HBM bound.
Confirmed independently: decode gained nothing from card6's +18% memory bandwidth while
prefill gained +13% from its core clock.

Refuted by measurement, not argument:
- **Occupancy / VGPR diet** — the *lower*-occupancy variant is faster (63 VGPR / 4 waves →
  26.01 vs 47 VGPR / 5 waves → 24.57). llama.cpp issue #20848's pathology does not transfer.
- **LDS → ALU** sign expansion — 20% slower, despite ~18% VALU utilization.
- **Byte-indexed LUT** (256 entries, half the LDS ops) — 11% slower from bank conflicts.
  This explains why the shipped nibble LUT wins: 16 × 8 B = 128 B maps 1:1 onto bank pairs,
  making it exactly large enough to be **bank-conflict-free**.
- **T-MAC activation-side LUT** — dropped unbuilt; it adds LDS traffic, the actual bottleneck.

### Measurement lessons worth keeping
1. **Rank by critical path, not per-op totals.** Removing the 48 per-layer state gathers cut
   2.45 ms of GPU op time but only 0.37 ms of wall clock — those gathers read the KV cache,
   are independent of the current layer's activations, and already overlapped with matvec.
2. **Decode is 100% GPU-bound**: 0 ms of process CPU over 10 s of steady decode.
3. `llama-bench` with mmap has ±7 variance on this box; use `--mmap 0 -r 3` (±0.1) for
   anything under 5%.
4. The RADV dump interleaves backend IR with final ISA; only ISA lines carry a `; hexcode`
   suffix. Counting IR as ISA reports max VGPR 4 — obvious nonsense, easy to believe.

## 5g. HBM overdrive validation (2026-08-02)

The rig now boots with `amdgpu.ppfeaturemask=0xffffffff`; the previous persistent GRUB
source and both generated configs have timestamped backups on the rig. On cards 1–3, with
core and memory DPM states explicitly pinned after each OD commit:

| HBM clock | prompt eval | decode | deterministic output |
|---|---:|---:|---|
| 800 MHz | 124.03 t/s | 30.93 t/s | oracle |
| 900 MHz | 124.64 t/s | **33.18 t/s** | byte-identical |

That is +0.5% prompt and **+7.3% decode**, without new AMDGPU errors. This establishes HBM
clock as a real decode lever despite the earlier cross-card Vega 64 comparison failing to
show it; that comparison also changed BIOS/core characteristics and was not a controlled
memory-clock A/B.

Two harness details are mandatory on this Vega driver. `pp_od_clk_voltage` requires the
unchanged 950 mV field (`m 3 <MHz> 950`), and committing the table silently resets DPM
selection. The checker therefore re-pins SCLK state 7 and MCLK state 3 after every commit,
including its cleanup back to 800 MHz. The first run without that re-pin produced an invalid
apparent regression (30.13 → 17.51 t/s at core state 0); it is excluded from the result.

The next HBM step, 950 MHz, was byte-identical but did not clear noise: 32.43 t/s versus a
fresh 800 MHz baseline of 32.75 t/s (prompt 126.78 versus 124.64). Keep 900 MHz as the best
proven point.

A separate core-only check held HBM at 800 MHz and changed cards 1–3 from stock
1590/1200 mV to 1700/1200 mV. It was stable, produced byte-identical output, and raised the
short 30-token prompt rate from 123.73 to 130.30 t/s (+5.3%), but decode was flat/slightly
lower at 33.55 versus 33.82 t/s. This is evidence to test long-prompt prefill, not sufficient
evidence to deploy the core overclock. The cards advertise 1200 mV as their maximum; 1250 mV
is outside the driver's OD range and was not attempted.

The production-sized prefill gate then falsified that short-prompt signal. At 1590 MHz,
pp2048 was 578.09 ± 1.44 and pp8192 was 524.39 ± 2.51. At 1700 MHz they were 583.70 ± 2.15
(+1.0%) and 527.82 ± 3.43 (+0.7%), respectively—inside variance and far below the 6.9% clock
increase. Combined with flat decode, 1700 MHz adds power without useful Qwen throughput and
should not be deployed.

## 5h. Four-card Qwen software/pipeline tuning (2026-08-02)

The real 5,629-token top-8 workload proves that the four-card bottleneck is not clock speed.
At the default queue/split, 800→900 MHz HBM changed 528.72/30.65 PP/TG to 530.64/30.76
(+0.36% on both). With graphics queue it changed 529.90/34.04 to 532.59/33.43 (+0.5% PP,
-1.8% TG). Every output was byte-identical; HBM was restored to 800 MHz.

Software placement did move the result materially:

| configuration | PP | TG | output |
|---|---:|---:|---|
| default queue and split | 528.72 | 30.65 | oracle |
| graphics queue | 529.67 | 35.47 | identical |
| graphics + `-ts 0.85,1.05,1.05,1.05` | **559.74** | **35.56** | identical |
| graphics + `-ts 0.70,1.10,1.10,1.10` | 560.09 | 33.36 | identical |
| first custom split + top-6 | **608.61** | 33.66 | semantics changed |

The first custom split is the top-8 Pareto winner: +5.7% PP over graphics/default split
without losing decode. Forward device order is mandatory; reversing `0,1,2,3` to `3,2,1,0`
collapsed performance to 88.55 PP / 2.81 TG. `-ub 1024` became pathologically slow after
normal model load and was terminated; retain 512.

The research-led backport of upstream PR #25862 (`5cea9089`, high-expert-count vec-ID path)
was also neutral at 528.56 PP versus 528.72 control, with identical output, and was removed.
The next implementation step is pipeline-safe boundary/kernel tracing, not another blind
tile sweep; see `docs/qwen4-optimization-roadmap.md`.

## 5i. Pipeline-safe scheduler trace and copy-path adjudication (2026-08-03)

An opt-in host-side scheduler tracer was added because Vulkan's existing per-op logger changes
pipeline behavior. With tracing disabled, the winning four-card split measured 560.12 PP / 35.76
TG and remained byte-identical. Trace-on prefill-only runs measured 557.17–561.74 PP, so its PP
perturbation is small enough for boundary diagnosis; it is not suitable for TG ranking because a
full trace-on run reduced TG by about 8%.

The trace found 24 synchronous fallback copies and 46,129,200 bytes at each of the three
downstream device boundaries. Their GPU-staged copy legs totaled roughly 0.8 seconds within a
10.1-second prefill, setting an approximate 8% ceiling for copy-only work. Dedicated transfer
queues did not improve either PP (562.21 versus the same ~562 trace band) or copy timings.

A guarded direct copy between coherent host-visible mappings then provided a decisive negative
control. PP collapsed to 269.31, while boundary copy time rose to 2.86, 8.53, and 0.29 seconds
(~11.7 seconds total), versus roughly 0.26–0.29 seconds per boundary through staging. This is
consistent with very slow CPU reads through uncached/BAR mappings on the rig. The direct-copy
patch was removed; the diagnostic scheduler tracer remains available behind
`GGML_SCHED_CRITICAL_TRACE=1`.

The follow-up fixed-counter Vulkan shape trace is synchronization-free and did not perturb
prefill: 558.42 PP. Across devices, Q4_K prefill generated 198–242 `MUL_MAT_ID` calls per card,
all in the 257–512-token MM bucket. Q4_K vec calls appeared only in the 2–8-token tail (16–22
per card). Consequently, changing the vec/MM cutoff cannot improve the main prefill workload;
the next kernel candidate is deterministic row-ID precompaction for the MM path.

That precompaction candidate was subsequently exhausted. The correctness-first implementation
used a stable flattened-order shared prefix scan, produced byte-identical output, but regressed
prefill to 549.02 PP. Replacing its preparation scan with the same subgroup-ballot ordering model
already used by the production MM-ID shader restored 559.22 PP with identical output. This is
only +0.14% over the 558.42 shape-trace control and does not clear noise or the 2% adoption gate.
The candidate and its roughly 4 MiB/card per-ubatch scratch allocation were removed. Repeated
routing scans are therefore not a material four-card bottleneck on this workload.

Upstream PR #24720's Vulkan command-buffer cache was then ported behind
`GGML_VK_GRAPH_REUSE=1`, with its unsafe timed eviction removed and aggregate lifecycle counters
added. The mechanism did activate: every card recorded 13 warmups, two captures, 123 replays,
and one invalidation during the 128-token workload. It nevertheless failed both performance and
correctness. The same binary measured 559.23 PP / 35.46 TG with reuse off and 559.38 / 33.81 with
reuse on (TG -4.7%). More importantly, deterministic output diverged after replay into repeated
multilingual garbage. The full graph-reuse series was removed without further PPL/server gates;
output corruption is already a decisive rejection.

The first isolated Mesa/ACO candidate was also rejected. Backporting the single-wave workgroup
barrier elimination (`49fb361c`) together with the required `a116cc91` execution-scope fix built
cleanly as a separate Mesa 26.1.5 runtime and retained byte-identical output. It nevertheless
regressed the production workload from 559.23 PP / 35.46 TG to 511.27 / 31.31: -8.6% PP and
-11.7% TG. The isolated Mesa target and patch were removed. On this shader mix, demoting/removing
those barriers is actively harmful rather than a small scheduling win.

The broader gfx9 RADV thread-ID/shuffle series (MR43022 plus MR43240 through `f44a6b57`, excluding
the gfx10-only tail) was then adapted to Mesa 26.1.5. Its bounded compatibility additions were the
MR-owned `BITSET_EXTRACT64` helper and the older NIR API's explicit component-count argument. The
full Mesa build and runtime completed, output remained byte-identical, but throughput dropped to
507.49 PP / 31.89 TG versus 559.23 / 35.46 control (-9.3%/-10.1%). The entire isolated override
was removed. Both credible post-26.1.5 gfx900 compiler candidates therefore regress this workload.
