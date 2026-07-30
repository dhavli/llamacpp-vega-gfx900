# Running Ternary-Bonsai-27B on a 7× Vega 56/64 Rig — Feasibility Analysis

Status: **VALIDATED ON HARDWARE 2026-07-30.** It runs. See `RESULTS.md` for measured
numbers and `benchmarks/llm/index.jsonl` for the full ledger.

The predictions below held up well, with three corrections marked inline as
**[CORRECTED]**. All claims are traced to source, measured third-party data, or our own
measurements; estimates are labelled as such.

Primary sources: the `prism` branch of `PrismML-Eng/llama.cpp` (cloned and inspected at commit
`7529fdaa`, 2026-07-19), rocBLAS/Tensile upstream sources, AMD ROCm docs, and the llama.cpp
Vulkan benchmark scoreboard.

---

## 1. Headline conclusion

**Vulkan is not a stepping stone to ROCm here — it is very probably the destination.**

The PrismML fork already contains a complete, non-cooperative-matrix Vulkan implementation of the
ternary `Q2_0` format *and* of the hybrid-attention op this model needs. Nothing needs to be
written to get this model running on Vega. The expected first step is "download the prebuilt
Vulkan binary and run it", not "port kernels".

The ROCm path, meanwhile, is buildable but its usual speed advantage largely evaporates on
gfx900 specifically, because gfx900 lacks every hardware feature ROCm's fast paths rely on
(see §5). ROCm should be treated as a *measured experiment*, not the plan.

---

## 2. Corrections to the working premise

These change the VRAM budget and the file to download, so they matter before anything else.

### 2.1 There is no 1-bit file *in this repo* — it lives in a different one

**[CORRECTED]** There *is* a 1-bit 27B: `prism-ml/Bonsai-27B-gguf` →
`Bonsai-27B-Q1_0.gguf`, **3.80 GB / 3.53 GiB**. The naming convention is the trap:

- `Ternary-Bonsai-*` = ternary, 2-bit slots (`Q2_0`), 2.125 bpw
- `Bonsai-*` (no prefix) = **binary 1-bit** (`Q1_0`), ±1 values, 1.125 bpw

I originally concluded the "~3.9 GB companion 1-bit model" was a different, smaller model.
It is not — it is this same 27B at Q1_0. **Q1_0 is the right target for 8 GB cards** and is
what all our measurements use.

The `prism-ml/Ternary-Bonsai-27B-gguf` repo contains:

| File | Size | Usable? |
|---|---|---|
| `Ternary-Bonsai-27B-Q2_0.gguf` | 7.17 GB | **yes — use this one** |
| `Ternary-Bonsai-27B-PQ2_0.gguf` | 7.17 GB | **no** — see §2.3 |
| `Ternary-Bonsai-27B-Q2_g64.gguf` | 7.59 GB | group-64 variant, larger, no benefit |
| `Ternary-Bonsai-27B-F16.gguf` | 53.8 GB | reference only |
| `Ternary-Bonsai-27B-dspark-Q4_1.gguf` | 1.95 GB | optional speculative drafter |
| `Ternary-Bonsai-27B-mmproj-Q8_0.gguf` | 629 MB | optional vision tower |

There is no `Q1_0` file. The "companion 1-bit model (~3.9 GB)" the model card recommends for
phones is a *different, smaller* model, not a 27B quant.

### 2.2 The real bit-width is 2.125 bpw, not 1.71

From `ggml/src/ggml-common.h`:

```c
typedef struct {
    ggml_half d;            // one fp16 scale
    uint8_t qs[QK2_0 / 4];  // 2 bits per weight
} block_q2_0;
```

128 weights → 2 bytes scale + 32 bytes payload = 34 bytes = **2.125 bits/weight on disk**.
27.3B × 2.125/8 ≈ 7.25 GB, which matches the 7.17 GB file. The advertised 1.71 bpw is an
information-theoretic figure, not the GGUF footprint. Budget VRAM from 7.17 GB.

### 2.3 `PQ2_0` is not supported by this fork

`grep -rn 'PQ2_0'` across `ggml/src`, `src`, and `tools` returns **zero hits**. Whatever that
file is, the current `prism` branch cannot load it. Do not download it.

### 2.4 It will not fit on one 8 GB card at 128k — true for Q2_0, FALSE for Q1_0

**[CORRECTED]** This section's arithmetic is correct *for Q2_0* (6.66 GiB of weights alone).
But with **Q1_0 (3.53 GiB), 128k context fits on one 8 GB Vega 56 with 1762 MiB to spare**
(measured peak 6414 MiB of 8176). Your original premise was right; I applied it to the wrong
file. Single-card deployment is viable and is now the recommended baseline.

Rough budget at 128k context:

| Component | f16 KV | q4_0 KV |
|---|---|---|
| Weights | 7.17 GB | 7.17 GB |
| KV cache (16 of 64 layers are full-attention) | ~2.1 GB | ~0.55 GB |
| Compute buffers / graph | ~0.8 GB | ~0.8 GB |
| **Total** | **~10.1 GB** | **~8.5 GB** |

Even with 4-bit KV this exceeds 8 GB. The model card's "fits on-device" numbers (~10.1 GB at
262k with 4-bit KV) describe a 12–16 GB card or a unified-memory Mac — not an 8 GB Vega.

**This is a non-issue on your rig.** 7 × 8 GB = 56 GB. Two cards with layer-split cover it
comfortably; see §6 for topology.

### 2.5 The model card understates backend support

The card says CUDA/Metal only. That is **false for the current `prism` branch**. The fork ships
prebuilt `bin-ubuntu-vulkan-x64` and `bin-ubuntu-rocm-7.2-x64` Linux binaries, and the Vulkan
source is complete (§3). Trust the source tree over the card.

---

## 3. What already exists on the Vulkan path

This is the key finding. Verified by inspection of `ggml/src/ggml-vulkan/`.

### 3.1 Ternary quant kernels — complete, and not coopmat-gated

`Q1_0` and `Q2_0` are wired into every pipeline the model needs, including the plain
scalar/fp32 variants that a GFX9 card will actually use:

- **Prompt processing:** `matmul_q2_0_f32` / `matmul_q2_0_f16` (`ggml-vulkan.cpp:4032`, `4128`,
  `4256`, `4429`) — registered in the non-coopmat and non-subgroup branches too.
- **Decode:** `mul_mat_vec_q2_0_f32_f32` and `..._f16_f32` (`:4600`, `:4627`).
- **Support ops:** `dequant_q2_0` (`:4737`), `get_rows_q2_0` (`:4765`, `:4793`),
  `cpy_f32_q2_0` (`:4877`), `set_rows_q2_0` (`:4890`).
- **Shader-level unpack:** `dequantize()` / `dequantize4()` / `get_dm()` for `DATA_A_Q2_0` in
  `vulkan-shaders/dequant_funcs.glsl:146` and `:558`.

The `Q2_0` unpack is about as cheap as a quant gets — a shift, a 2-bit mask, and a subtract:

```glsl
const uint byte_val = uint(data_a[a_offset + ib].qs[iqs / 4u]);
const uint shift = (iqs % 4u) * 2u;
// float(int((byte_val >> shift) & 3u) - 1)   →  {-1, 0, +1}
```

No lookup tables, no nibble juggling, no per-element scales. Good news for Vega's register
budget (§5.3).

Note `mul_mat_vecq_funcs.glsl` and `mul_mmq_funcs.glsl` have **no** `q2_0` entries — the
integer-dot (`q8_1`) path does not cover this type. Vulkan will use the fp16/fp32 dequant path
instead. On gfx900 that is *fortunate*, not a limitation: the card has no dp4a anyway (§5.1) but
does have double-rate fp16.

### 3.2 The hybrid-attention op is implemented on Vulkan

This was the make-or-break question, since ~75% of the backbone is linear attention.

`GGML_OP_GATED_DELTA_NET` has a real Vulkan shader — `vulkan-shaders/gated_delta_net.comp`,
dispatched at `ggml-vulkan.cpp:14234`, declared supported at `:16891`. `ssm_scan.comp` and
`ssm_conv.comp` exist too.

Constraints to be aware of (`ggml-vulkan.cpp:16891`):

```cpp
case GGML_OP_GATED_DELTA_NET:
    // rows-indexed state read (src[6]) not implemented on Vulkan yet
    if (op->src[6] != nullptr) return false;
    const uint32_t S_v = op->src[2]->ne[0];
    if (S_v != 32 && S_v != 64 && S_v != 128) return false;
    for (int i = 0; i < 6; i++)
        if (op->src[i] == nullptr || op->src[i]->type != GGML_TYPE_F32) return false;
    return op->type == GGML_TYPE_F32;
```

So: head value-dim must be 32/64/128, and no rows-indexed state read. **Verify the 27B's actual
`S_v` early** — if it is not in that set, this op silently falls back to CPU and destroys
throughput. This is the single most important thing to check in Phase 1.

### 3.3 Quantized KV cache with flash attention works

From the `GGML_OP_FLASH_ATTN_EXT` gate (`:16582`), the scalar (non-coopmat) FA path accepts
`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` for K and V. So `-ctk q4_0 -ctv q4_0` is available on
Vega, which is what makes long context affordable.

Two caveats:
- `GGML_TYPE_Q1_0` as a KV type requires `coopmat2` → NVIDIA-only. Irrelevant here.
- The scalar FA path requires `subgroup_shuffle` **and** `subgroup_vote`. RADV exposes both on
  GFX9, so Vega qualifies — but confirm in `vulkaninfo` (§7, Phase 0).

### 3.4 Net: nothing must be written to get this running

Every op this model needs has a Vega-compatible Vulkan implementation. The gap between "we have
7 Vega cards" and "the model is generating tokens" is build-and-configure work, not kernel
development.

---

## 4. What is missing / genuinely at risk

Ordered by how likely they are to bite.

1. **`S_v` mismatch in the delta-net gate** (§3.2). If the 27B's head dim isn't 32/64/128, the
   hybrid layers fall back to CPU. Highest-priority check. Fix would be extending the shader —
   moderate GLSL work, very much doable.
2. **Delta-net shader is untuned for wave64.** It exists, but nobody has optimised it for GFX9.
   Expect this to be a top-2 hotspot in profiling.
3. **`Q2_0` mul_mat_vec occupancy on GCN.** See §5.3 — this is the known failure mode for Vega
   in llama.cpp Vulkan, and the most likely source of easy wins.
4. **No coopmat on Vega** → prefill uses the scalar matmul path. Costs prompt-processing speed,
   not decode. Partly offset by the model being 75% linear attention, which scales far better
   than quadratic at long context.
5. **Vision tower (`mmproj`) on Vulkan** — not audited. If you need image input, treat it as a
   separate unknown; text-only works regardless.
6. **DSpark drafter on Vulkan** — the 1.34× speedup is a CUDA figure. Batch-1 verification
   overhead may eat it on Vega, and the drafter costs 1.95 GB. Test with and without.
7. **7-device Vulkan enumeration** is a lightly-travelled configuration. Expect to need
   `GGML_VK_VISIBLE_DEVICES` to pin a subset.

---

## 5. Vega 56/64 hardware reality — why ROCm's advantage mostly disappears

llama.cpp's own source documents Vega's position precisely
(`ggml/src/ggml-cuda/common.cuh:68-79`):

```c
#define GGML_CUDA_CC_GCN4    (…+ 0x803)  // Tonga, Fiji, Polaris, minimum for fast fp16
#define GGML_CUDA_CC_VEGA    (…+ 0x900)  // Vega56/64, minimum for fp16 dual issue
#define GGML_CUDA_CC_VEGA20  (…+ 0x906)  // MI50/Radeon VII, minimum for dp4a
#define GGML_CUDA_CC_CDNA1   (…+ 0x908)  // MI100, minimum for MFMA, acc registers
```

### 5.1 gfx900 has no dp4a

Your cards sit **one tier below the dp4a minimum**. gfx906 (Radeon VII / MI50) has
`v_dot4_i32_i8`; gfx900 does not. llama.cpp works around this with hand-written inline asm
(`common.cuh:694`):

```c
#elif defined(RDNA1) || defined(__gfx900__)
    // 4× v_mul_i32_i24 with SDWA byte selects + 2× v_add3_u32
```

That is ~6 instructions to replace 1. Integer quantized matmul (MMQ) *works* on gfx900, but at
roughly a fifth of the per-instruction dot throughput of gfx906.

This one fact explains the benchmark cliff between Vega 64 and Radeon VII in §5.4 — they have
near-identical memory bandwidth, yet the VII is ~1.6× faster.

### 5.2 No MFMA, no WMMA, no coopmat

- No matrix cores (MFMA starts at CDNA1/gfx908).
- No WMMA (RDNA3+).
- RADV exposes `VK_KHR_cooperative_matrix` only on **GFX11 (RDNA3) and newer** — it is
  implemented on top of WMMA instructions Vega does not have. Vega will never get coopmat on RADV.

What Vega *does* have is **double-rate packed fp16** (25.3 TFLOPS on Vega 64 vs 12.7 fp32),
which is exactly what the Vulkan fp16 dequant-matmul path uses. This is the strength to lean on.

### 5.3 The known Vega occupancy trap

llama.cpp issue #20848 documents IQ3 dequant shaders in `mul_mat_vec.comp` compiling to
**64 VGPRs** under RADV's ACO on GCN, capping occupancy at 4 waves/SIMD (40%) and starving HBM2
bandwidth. Simpler formats do far better — Q8_0 reaches 229 GB/s effective by combining vec4
loads with 60% occupancy.

`Q2_0`'s unpack is *simpler* than Q4_0's, so it should land on the good side of this — but this
is the first thing to verify with `RADV_DEBUG=asm` and the most likely source of a cheap 20–40%
win. Target ≤32 VGPRs.

### 5.4 Measured baseline (llama.cpp Vulkan scoreboard, RADV, Llama-2-7B Q4_0)

| GPU | pp512 (t/s) | tg128 (t/s) | Bandwidth |
|---|---|---|---|
| Vega 56 | 439.9 | 50.2 | 410 GB/s |
| **Vega 64** | **575.1** | **63.4** | **484 GB/s** |
| Radeon VII (gfx906) | 1059.1 | 101.2 | 1024 GB/s |
| MI50 (gfx906) | 1119.6 | 108.5 | 1024 GB/s |

Vega 64 at 63.4 t/s on a 3.83 GB model → **~240 GB/s effective decode bandwidth**, about 50% of
peak. That is a healthy, well-trodden result: Vulkan on Vega 10 is not a broken path.

### 5.5 Projected performance for this model (estimate)

Decode is weight-bandwidth-bound. At ~240 GB/s effective over 7.17 GB of weights:

- **Ceiling: ~33 tok/s.**
- **Realistic: 15–30 tok/s**, after the delta-net recurrent state work, an untuned hybrid-attention
  shader, and layer-split sync overhead. Vega 56 ≈ 0.85× that.
- Layer-splitting across 2 cards does *not* halve this — GPUs run sequentially, so aggregate time
  ≈ total bytes ÷ per-GPU bandwidth. It buys capacity, not speed.

Prefill is compute-bound and will be the weak spot (no coopmat, no dp4a) — order 100–250 t/s.
The 75%-linear-attention backbone helps disproportionately at long context, which is precisely
the regime you care about.

**This is a usable interactive speed.** For comparison the model card quotes 26 tok/s on an
M5 Pro laptop.

---

## 6. The ROCm question, answered precisely

### 6.1 The blocker chain

`ggml/src/ggml-hip/CMakeLists.txt`:

```cmake
find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()
```

So llama.cpp needs **ROCm ≥ 6.1** with rocBLAS and hipBLAS present. Conventional wisdom says
gfx900 died at ROCm 4.5.2, which would make this impossible. **Conventional wisdom is wrong.**

### 6.2 gfx900 is still a buildable rocBLAS target — through ROCm 7.13

From rocBLAS `develop` `CMakeLists.txt:89-95` (fetched live):

```cmake
set( TARGET_LIST_ROCM_6.0  "gfx900;gfx906:xnack-;…")
set( TARGET_LIST_ROCM_6.3  "gfx900;gfx906:xnack-;…")
set( TARGET_LIST_ROCM_7.0  "gfx900;gfx906:xnack-;…")
set( TARGET_LIST_ROCM_7.1  "gfx900;gfx906:xnack-;…")
set( TARGET_LIST_ROCM_7.13 "gfx900;gfx906:xnack-;…")
```

`gfx900` is in **every** list up to 7.13. Tensile agrees — `Tensile/Common.py` still lists ISA
`(9,0,0)` and maps `'gfx900':'vega10'`. And AMD's *own* llama.cpp-on-ROCm documentation for
**ROCm 7.0.0** gives this as a supported arch list:

```
gfx803,gfx900,gfx906,gfx908,gfx90a,gfx942,gfx1010,gfx1030,gfx1032,gfx1100,gfx1101,gfx1102
```

**The gap is packaging, not code.** gfx900 fell off AMD's *support matrix* and may be missing
from shipped binaries (hence the widely-reported missing `TensileLibrary_lazy_gfx900.dat`), but
it remains a first-class build target in source. Rebuild rocBLAS with `-DGPU_TARGETS=gfx900`
and you have it.

### 6.3 Do NOT set HSA_OVERRIDE_GFX_VERSION

Much of the internet's gfx900 despair — including "the new modular ROCr rejects
`HSA_OVERRIDE_GFX_VERSION=9.0.0` with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`" — concerns Vega
**APUs** (`gfx90c`, Vega 8) *impersonating* gfx900.

Vega 56/64 are **genuine gfx900 discrete GPUs**. They need no override at all. Setting one will
only cause problems. This distinction is the difference between a hard blocker and a non-issue,
and most guides conflate them.

Likewise, `bajceta/rocm_gfx900_pytorch` (pinning ROCm 5.4.3 via `xuhuisheng/rocm-build`) solves
a *different* problem — full-stack source builds including MIOpen/rocSPARSE/RCCL for PyTorch.
llama.cpp needs only hip + rocBLAS + hipBLAS: no MIOpen, no composable_kernel, no RCCL. Your
build is far smaller than that repo suggests, and does not require dropping to ROCm 5.x.

### 6.4 What ROCm would actually buy you on gfx900

MMQ *is* reachable on gfx900. Tracing `ggml_cuda_should_use_mmq` (`mmq.cu:270`): `Q2_0` is in
the supported-type list; the `< GGML_CUDA_CC_DP4A` gate compares against 610 while AMD codes
start at `0x1000000`, so all AMD parts pass; gfx900 is not MFMA/WMMA/CDNA, so it reaches
`return (!GGML_CUDA_CC_IS_CDNA(cc)) || …` → **true**.

So MMQ runs — on the 6-instruction emulated dp4a. Meanwhile hipBLAS's gfx900 Tensile kernels are
unmaintained and untuned for a 2017 part.

**Honest verdict:** ROCm's reputation for beating Vulkan on AMD rests on dp4a, MFMA/WMMA, and a
well-tuned hipBLAS. gfx900 has none of the first three, and the fourth is bit-rotted. Meanwhile
Vulkan/RADV on Vega 10 is actively exercised and the fp16 path suits the hardware. Historically,
llama.cpp on Vega 10 has generally shown **Vulkan ≥ ROCm**.

Treat ROCm as a Phase-4 experiment with a clear kill criterion, not as the goal.

---

## 7. Recommended path forward

### Phase 0 — Environment audit (hours, no downloads)

Per card, confirm: driver is **RADV/mesa** (not AMDVLK — llama.cpp is better tested on RADV and
ACO generates better GCN code); mesa reasonably recent; and that `vulkaninfo` reports
`shaderFloat16`, `subgroupShuffle`, `subgroupVote`, `subgroupBallot`, subgroup size 64.
Also record PCIe link width per slot — `lspci -vv | grep -E 'LnkCap|LnkSta'`. On a mining rig
expect x1 risers; you need to know before choosing a split mode.

### Phase 1 — Smoke test with prebuilt binaries (day)

Highest value per unit effort, and it may simply work:

1. Fetch `llama-prism-<ver>-bin-ubuntu-vulkan-x64.tar.gz` from the fork's releases — no compiling.
2. Download **`Ternary-Bonsai-27B-Q2_0.gguf`** (7.17 GB). Not `PQ2_0`.
3. Run text-only on **two** cards, layer split, 4-bit KV:
   `-ngl 99 --split-mode layer -ctk q4_0 -ctv q4_0 -c 32768`
4. **Immediately check for CPU fallback of the hybrid layers** (§3.2). Any op running on CPU
   makes the whole exercise pointless. Confirm `S_v ∈ {32,64,128}` and that
   `GGML_OP_GATED_DELTA_NET` is on the Vulkan device.
5. Only then push context to 128k.

### Phase 2 — Source build + baseline (day)

Build the `prism` branch with `-DGGML_VULKAN=ON` so you can patch. Record `llama-bench` numbers
for pp/tg at several context lengths. Profile to find real hotspots (`GGML_VULKAN_PERF`,
`RADV_DEBUG`). Do not optimise before this data exists.

### Phase 3 — Vulkan optimisation (the real work)

In expected order of payoff:
1. **`Q2_0` mul_mat_vec occupancy** — `RADV_DEBUG=asm`, count VGPRs, target ≤32 (§5.3). Cheapest
   likely win.
2. **Wave64 tuning of `gated_delta_net.comp`** — workgroup/subgroup sizing for GFX9.
3. Subgroup-size and workgroup tuning for the scalar matmul path.
4. Evaluate the DSpark drafter; keep only if it pays for its 1.95 GB.

Items 1 and 2 are plausibly **upstreamable to llama.cpp** — issue #20848 is exactly this class
of problem, and a Vega-tuned ternary path benefits every GCN user.

### Phase 4 — ROCm experiment (optional, only if Phase 3 disappoints)

1. ROCm 6.4 or 7.x userspace (kernel `amdgpu` already supports Vega 10).
2. Build rocBLAS from source: `-DGPU_TARGETS=gfx900`. Skip MIOpen/RCCL/composable_kernel.
3. Build the fork with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx900`, try `GGML_CUDA_FORCE_MMQ=ON`.
4. Do **not** set `HSA_OVERRIDE_GFX_VERSION` (§6.3).
5. **Kill criterion:** if it is not clearly faster than tuned Vulkan, drop it. Given §6.4, a
   loss is the likely outcome.

---

## 8. Rig-level notes

- **Topology.** ~9–10 GB needed at 128k → 2 cards minimum. With 7 cards, the throughput-optimal
  layout is probably **3 independent 2-card server instances** (3 concurrent sessions, ~3× aggregate
  tokens/s) with 1 card spare for the drafter/vision tower — rather than one 7-way split, which
  adds sync overhead without adding speed (§5.5).
- **Never use `--split-mode row` on x1 risers.** Row split streams partial results across PCIe
  every layer. Layer split only passes activations at layer boundaries (a few hundred KB) and is
  fine even on x1.
- **Load time** over x1 risers will be slow; rely on page cache / `--mlock` for restarts.
- **Power.** 7 Vega 56/64 is ~1.5–2 kW under load. Decode is bandwidth-bound, so power-capping
  and undervolting cost very little throughput — cap them. Prioritise memory clock over core
  clock; HBM2 bandwidth is the thing that matters.
- **Vega HBCC** (extending VRAM into system RAM) is sometimes floated as a way past the 8 GB
  limit. It is primarily a Windows Radeon-Settings feature with weak Linux support; do not build
  the plan around it. Multi-GPU split is the sane answer.

---

## 9. Open questions to resolve first

1. What is the 27B's delta-net `S_v` (head value dim)? Must be 32, 64, or 128 (§3.2). **Blocking.**
2. Does the model's graph ever set `src[6]` on `GATED_DELTA_NET` (rows-indexed state read)? Vulkan
   rejects that.
3. What is `Ternary-Bonsai-27B-PQ2_0.gguf`? Unsupported by this fork commit — possibly a newer
   packed format worth tracking.
4. Is the `mmproj` vision path Vulkan-clean? Only matters if you need images.
5. Does the fork's ROCm 7.2 prebuilt binary include gfx900 kernels? If yes, Phase 4 gets much
   cheaper — worth checking the shipped `TensileLibrary*` files before building anything.

---

## 10. Sources

- [prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) — model card, file list
- [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) — `prism` branch, inspected at `7529fdaa`
- [PrismML llama.cpp docs](https://docs.prismml.com/run/llamacpp) — backend build flags
- [llama.cpp Vulkan performance scoreboard (#10879)](https://github.com/ggml-org/llama.cpp/discussions/10879) — Vega 56/64 measurements
- [llama.cpp issue #20848](https://github.com/ggml-org/llama.cpp/issues/20848) — GCN/Vega VGPR occupancy problem
- [ROCm/rocBLAS `CMakeLists.txt`](https://github.com/ROCm/rocBLAS/blob/develop/CMakeLists.txt) — gfx900 in target lists through ROCm 7.13
- [AMD: llama.cpp on ROCm installation](https://rocm.docs.amd.com/projects/llama-cpp/en/docs-26.02/install/llama-cpp-install.html) — gfx900 in ROCm 7.0 arch list
- [RADV cooperative matrix is RDNA3+ only](https://www.phoronix.com/news/RADV-VK_KHR_cooperative_matrix)
- [bajceta/rocm_gfx900_pytorch](https://github.com/bajceta/rocm_gfx900_pytorch) and [xuhuisheng/rocm-build](https://github.com/xuhuisheng/rocm-build) — full-stack source builds (more than llama.cpp needs)
- [ROCm/TheRock #2588](https://github.com/ROCm/TheRock/issues/2588) — "gfx900 is functional, it just needs a few fixes"
