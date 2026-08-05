# Comprehensive Technical Analysis & Kernel Optimization Notes

Kernel work, architectural deep-dive, measurements, and technical analysis from running LLMs on **AMD Radeon RX Vega 56/64** via `llama.cpp`'s **Vulkan** backend (Mesa RADV + ACO).

---

## 1. Hardware Context & Microarchitecture Reality (gfx900 / GCN5)

AMD Vega 56 and Vega 64 GPUs (gfx900 architecture) represent high memory-bandwidth hardware (410–484 GB/s HBM2) without modern matrix hardware:
- **No dp4a / no `int8` dot product**: `v_dot4_i32_i8` starts at gfx906 (Radeon VII / MI50). On gfx900, integer dot products require ~6 SDWA emulate instructions per 4 bytes.
- **No matrix cores / no coopmat**: `VK_KHR_cooperative_matrix` is RDNA3+ (GFX11) only in Mesa RADV.
- **FP16 Rate**: `v_fma_f16` runs at scalar fp32 rate unless ACO emits packed `v_pk_fma_f16` (25.3 TFLOPS on Vega 64 vs 12.7 TFLOPS fp32).

Because packed-fp16 is the primary double-rate compute engine on Vega 10, Vulkan fp16/q8 dequant paths outperform ROCm paths.

---

## 2. Multi-GPU Pipeline Dynamics on PCIe Gen2 x1 Risers

On mining rigs with PCIe Gen2 x1 risers (~0.5 GB/s PCIe bandwidth per riser):
1. **Layer-Splitting Across Cards (Pipeline Parallelism)**:
   - Prefill scales with prompt length because ubatches are submitted asynchronously across cards.
   - Decode has no sequence overlap: every layer boundary incurs a blocking cross-device copy via host-staged `ggml_vk_buffer_copy` (~3.6 ms per boundary).
   - Dense models (e.g., 27B) split across cards lose decode speed as card count grows beyond 3–4 cards.
2. **Sparse MoE Architecture Advantage (Mellum2 12B MoE)**:
   - Mellum2 12B MoE has 12B total / 2.5B active parameters.
   - Because only 2.5B parameters are active per token, 100% of the model weights (7.03 GB) and 128k Q8 KV cache fit on a **SINGLE 8 GB Vega 56 card**.
   - Running 7 independent single-card instances eliminates PCIe cross-card boundary traffic entirely during decode, achieving **63.3 TG tok/s per GPU** (440+ TG tok/s aggregate).

---

## 3. Vulkan Kernel Work & Patch Breakdown

`patches/0001-vulkan-wide-ternary-mmv-gcn-knobs.patch` contains Vulkan kernel optimizations for GCN5/Vega:

- **`PACKED_K` mul_mm** (`GGML_VK_MM_PACKED=1`): Pairs even/odd k into `f16vec2` lanes so ACO emits `v_pk_fma_f16` instead of scalar `v_fma_f16`. **+33% prefill gain.**
- **LDS Sign-Nibble LUT Matvec** (`GGML_VK_TERNARY_LUT=1`): Maps 4 sign bits to `f16vec4(±1)` through a 16-entry shared table (`ds_read_b64` + 2 packed FMAs per 8 weights). **+24% decode gain.**
- **Software-Pipelined Matvec Prefetch** (`GGML_VK_TERNARY_PREFETCH=1`): Hoists per-row weight loads ahead of FMA chains. **+5.7% decode gain.**
- **GCN VGPR Occupancy Tuning**: IQ3 dequant shaders under ACO can compile to 64 VGPRs (capping occupancy at 40%). Simplifying unpack logic to $\le 32$ VGPRs restores 60%+ CU occupancy.

---

## 4. Single-GPU MTP Speculative Decoding Analysis

MTP (Multi-Token Prediction) speculative decoding for Gemma 4 12B (`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, 6.71 GB) with `mtp-gemma-4-12B-it-Q8_0.gguf` (465 MB):
- On a **single GPU**, MTP draft evaluation runs entirely in local VRAM without PCIe riser latency.
- Optimal draft depth is **`draft-n-max 3`**, achieving **25.91 TG tok/s (+47.6% speedup)** over solo execution (17.55 TG tok/s) at **73.7% draft acceptance**.
- Higher draft depths (`n_max 4`, `n_max 5`) introduce diminishing returns due to draft verification overhead on 12B parameters.

---

## 5. Summary of Negative Results & Dead Ends

- **ROCm on gfx900**: Tested against Vulkan. On Vega 10 without dp4a or matrix cores, ROCm hipBLAS kernels suffer from bit-rot on GCN5; Vulkan/RADV ACO delivers equal or superior throughput with better driver stability.
- **Unvalidated Warptiles**: `GGML_VK_TILE_M` overrides that appeared to triple prefill speed were found to skip matmul compute blocks, outputting garbage tokens while passing synthetic benchmark loops. All warptile tuning must be verified against deterministic greedy output SHA-256 hashes.
- **P2P / Cross-Card Copying**: Vulkan backend lacks `VK_KHR_device_group` P2P memory access on RADV for GCN5; cross-card layer boundaries must stage through host RAM fences.
