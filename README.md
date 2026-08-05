# llama.cpp Vulkan Guide for AMD Radeon RX Vega 56/64 (gfx900)

A straightforward guide and benchmark reference for running modern LLMs (**JetBrains Mellum2 12B MoE**, **Gemma 4 12B**, and **Bonsai 27B**) on a **single 8 GB AMD Radeon RX Vega 56/64 GPU** using `llama.cpp` with the **Vulkan** backend (Mesa RADV + ACO).

---

## 🚀 Tested Models & Best Performance Matrix (Single Vega 56/64 GPU)

| Model & Quantization | Active Params | Context / KV Quant | Prefill (PP) | Decode (TG) | Optimal Runtime Configuration | Key Takeaway |
|---|:---:|:---:|:---:|:---:|---|---|
| **JetBrains Mellum2 12B MoE** (`MXFP4_MOE`, 7.03 GB) | **2.5B Active** / 12B Total | **128k (`Q8_0` KV)** | **750.4 tok/s** | **61.1 tok/s** (68.3 peak) | `-ngl 99 -fa on --no-mmap -c 131072 -ctk q8_0 -ctv q8_0 -b 1408 -ub 384` | **Best Overall Model**: Full 128k context in VRAM @ 61 TG/s |
| **Gemma 4 12B Q4_K_XL** + **Q8 MTP Drafter** | 12B Dense | 16k (`Q8_0` KV) | **230.2 tok/s** | **25.9 tok/s** | `-md mtp-gemma-4-12B-it-Q8_0.gguf -draft-n-max 3 -ngl 99 -fa on --no-mmap` | **+47.6% Decode Speedup** via local single-GPU MTP drafting |
| **Gemma 4 12B Q4_K_XL** (Solo) | 12B Dense | 16k (`Q8_0` KV) | **233.3 tok/s** | **17.6 tok/s** | `-ngl 99 -fa on --no-mmap -c 16384 -ctk q8_0 -ctv q8_0 -b 1408 -ub 384` | Baseline solo 12B dense execution |
| **Bonsai 27B Q1_0** (`Q1_0`, 3.53 GiB) | 27B Dense | 32k (`Q4_0` KV) | **206.4 tok/s** (225.9 V64) | **26.1 tok/s** | `GGML_VK_TERNARY_LUT=1 GGML_VK_MM_PACKED=1 GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | 1-bit ternary architecture with custom LDS LUT kernels |

---

## 🛠️ Model Configurations & How to Run

### 1. JetBrains Mellum2 12B MoE (Best Production Choice)
Fits **100% inside 8 GB VRAM** with **128k context and Q8 KV cache** (~7.51 GB VRAM / ~92% occupancy). Uses **~75 MiB host RAM** via `--no-mmap`.

```bash
# Model download:
# Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf (7.03 GB)

GGML_VK_VISIBLE_DEVICES=0 \
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
RADV_PERFTEST=nogttspill \
LLAMA_SERVER_OUTPUT_RESERVE=128 \
llama-server \
  -m Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf \
  -ngl 99 \
  -fa on \
  --no-mmap \
  -c 131072 \
  -np 1 \
  -ctk q8_0 \
  -ctv q8_0 \
  -b 1408 \
  -ub 384 \
  --cache-ram 0 \
  --ctx-checkpoints 0 \
  --host 0.0.0.0 \
  --port 8080
```

---

### 2. Gemma 4 12B with MTP Speculative Decoding (+47.6% Speedup)
Using the official Q8 MTP drafter on a single card eliminates PCIe latency and boosts decode speed from 17.6 to **25.9 TG tok/s**.

```bash
# Models:
# Main: gemma-4-12B-it-qat-UD-Q4_K_XL.gguf (6.71 GB)
# Drafter: mtp-gemma-4-12B-it-Q8_0.gguf (465 MB)

GGML_VK_VISIBLE_DEVICES=0 \
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
RADV_PERFTEST=nogttspill \
llama-server \
  -m gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  -md mtp-gemma-4-12B-it-Q8_0.gguf \
  -draft-n-max 3 \
  -ngl 99 \
  -fa on \
  --no-mmap \
  -c 16384 \
  -np 1 \
  -ctk q8_0 \
  -ctv q8_0 \
  -b 1408 \
  -ub 384
```

---

### 3. Bonsai 27B Q1_0 (1-Bit Ternary Architecture)
Extremely compact 27B model (3.53 GiB) fitting easily on a single 8 GB card with 32k context.

```bash
# Model: Bonsai-27B-Q1_0.gguf (3.80 GB / 3.53 GiB)

GGML_VK_VISIBLE_DEVICES=0 \
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
GGML_VK_MM_PACKED=1 \
GGML_VK_TERNARY_LUT=1 \
GGML_VK_TERNARY_PREFETCH=1 \
GGML_VK_TERNARY_SOA=1 \
GGML_VK_TERNARY_ROWS=8 \
GGML_VK_GCN_SUBGROUP_REDUCE=1 \
llama-server \
  -m Bonsai-27B-Q1_0.gguf \
  -ngl 99 \
  -fa on \
  --no-mmap \
  -c 32768 \
  -ctk q4_0 \
  -ctv q4_0 \
  --host 0.0.0.0 \
  --port 8080
```

---

## 📊 Comprehensive Context Scaling Matrix (Mellum2 12B MoE)

| Context Allocation (`-c`) | Prompt Processing (PP) | Text Generation (TG) | Single-Card VRAM Occupancy | SHA-256 Match |
|---|:---:|:---:|---|:---:|
| **16k (`16384`)** | 748.6 tok/s | 61.4 tok/s | ~90% VRAM Resident | `38e0b9de81...` |
| **32k (`32768`)** | 747.4 tok/s | 61.5 tok/s | ~91% VRAM Resident | `38e0b9de81...` |
| **64k (`65536`)** | 884.7 tok/s | 64.0 tok/s | ~91.5% VRAM Resident | `38e0b9de81...` |
| **128k (`131072`)** | **750.4 tok/s** | **61.1 tok/s** | **~92% VRAM Resident (7.51 GB)** | `38e0b9de81...` |
| **176k (`180224`)** | **751.3 tok/s** | **59.9 tok/s** | **~97% VRAM Resident** | `38e0b9de81...` |
| **224k (`229376`)** | **674.2 tok/s** | **14.9 tok/s** | **~99% VRAM Upper Limit** | `38e0b9de81...` |
| **256k (`262144`)** | 92.8 tok/s | 0.9 tok/s | 99.9% VRAM GTT Thrashing Cliff | `38e0b9de81...` |

---

## 🌐 Multi-GPU Cluster Setup (7x Vega 56 for Bifrost Proxy)

To run a 7-GPU cluster where each card runs an independent 128k context instance:

```bash
# Starts 7 llama-server backends on ports 8001..8007 (GPUs 0..6)
./benchmarks/llm/launch-7gpu-mellum2-cluster.sh
```

Bifrost proxy can route requests directly to:
- `http://0.0.0.0:8001/v1` (GPU 0)
- `http://0.0.0.0:8002/v1` (GPU 1)
- ...
- `http://0.0.0.0:8007/v1` (GPU 6)

**Aggregate Cluster Throughput**: **~440–470 TG tok/s** across 7 concurrent streams with **887 MiB free host RAM** on a 3.8 GB system.

---

## 📚 Technical Documentation & Deep-Dive Links
- [ANALYSIS.md](file:///home/orchestra/orca/projects/vega-bonsai/ANALYSIS.md): Comprehensive microarchitecture deep-dive, GCN occupancy traps, Vulkan kernel optimization work, and multi-GPU pipeline analysis.
- [RESULTS.md](file:///home/orchestra/orca/projects/vega-bonsai/RESULTS.md): Chronological experiment log and raw benchmark ledger.
- [patches/](file:///home/orchestra/orca/projects/vega-bonsai/patches): Custom Vulkan GCN kernel patches (`0001-vulkan-wide-ternary-mmv-gcn-knobs.patch`).
