# Bonsai-27B on Vega 56 — Measured Baseline (2026-07-30)

**It works.** A 27B vision-language model with 128k context, running on a single 2017
8 GB Vega 56 via Vulkan, at ~13.4 tok/s. No code was written — prebuilt binary, stock RADV.

Full ledger: `benchmarks/llm/index.jsonl` (12 records, including invalid/artifact runs).

---

## 1. Environment

| | |
|---|---|
| Host | `RIG4AMDRX580Vega` (100.95.237.97), HiveOS / Ubuntu 22.04.5, kernel 6.6.0-hiveos |
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
