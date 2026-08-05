# llama.cpp Vulkan Guide for AMD Radeon RX Vega 56/64 (gfx900)

A straightforward guide and benchmark reference for running modern LLMs (**JetBrains Mellum2 12B MoE** and **Gemma 4 12B**) on a **single 8 GB AMD Radeon RX Vega 56/64 GPU** using `llama.cpp` with the **Vulkan** backend (Mesa RADV + ACO).

---

## 🚀 Key Benchmarks (Single Vega 56 GPU, 8 GB VRAM)

### 1. JetBrains Mellum2 12B-A2.5B MoE (`MXFP4_MOE`)
- **Model**: `Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf` (7.03 GB, 12B total / 2.5B active MoE).
- **VRAM Footprint**: **Fits 100% in 8 GB VRAM** with **128k context & Q8 KV cache** (~7.51 GB VRAM / ~92% occupancy).
- **Host RAM Footprint**: Uses **~75 MiB host RAM** via `--no-mmap` (runs easily on low-RAM machines).
- **Performance**:
  - **128k Context (`-c 131072`)**: **750.4 PP tok/s**, **61.1 TG tok/s**!
  - **176k Context (`-c 180224`)**: **751.3 PP tok/s**, **59.9 TG tok/s**!
  - **224k Context (`-c 229376`)**: **674.2 PP tok/s**, **14.9 TG tok/s** (Upper VRAM limit before 256k GTT thrashing).

### 2. Gemma 4 12B Q4_K_XL (Solo vs Q8 Speculative Drafter)
- **Model**: `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` (6.71 GB) + MTP Drafter `mtp-gemma-4-12B-it-Q8_0.gguf` (465 MB).
- **Performance**:
  - **Solo Execution**: **233.3 PP tok/s**, **17.6 TG tok/s**.
  - **Q8 MTP Drafter (`n_max=3`)**: **230.2 PP tok/s**, **25.9 TG tok/s (+47.6% decode speedup)** at 73.7% draft acceptance!

---

## 🛠️ Quickstart: How to Run Mellum2 12B on a Single Vega GPU

### Prerequisites
- AMD Radeon RX Vega 56 or Vega 64 (8 GB VRAM)
- `llama.cpp` built with Vulkan support (`-DGGML_VULKAN=ON`)
- Mesa RADV graphics driver (`vulkan-radeon`)

### Step 1: Download Model Weights
```bash
python3 -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-MXFP4_MOE/resolve/main/Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf', 'Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf')"
```

### Step 2: Run `llama-server` (Single Card, 128k Context)
```bash
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

### Critical Flags Explained
- `GGML_VK_VISIBLE_DEVICES=0`: Binds execution strictly to GPU 0.
- `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`: Enables graphics queue submission (+18% decode throughput on Vulkan).
- `RADV_PERFTEST=nogttspill`: Prevents RADV from spilling VRAM allocations to system GTT memory.
- `-ngl 99`: Offloads all 42 transformer layers into GPU VRAM.
- `--no-mmap`: Disables memory-mapping model files into host RAM, keeping host memory usage at ~75 MiB.
- `-c 131072 -ctk q8_0 -ctv q8_0`: Allocates 128k context with 8-bit quantized KV cache inside GPU VRAM.
- `--cache-ram 0`: Disables host RAM prompt caching to preserve system memory.

---

## ⚡ How to Run Gemma 4 12B with MTP Speculative Decoding

### Solo Execution (Single Card)
```bash
GGML_VK_VISIBLE_DEVICES=0 GGML_VK_ALLOW_GRAPHICS_QUEUE=1 RADV_PERFTEST=nogttspill \
llama-server \
  -m gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on --no-mmap -c 16384 -np 1 -ctk q8_0 -ctv q8_0 -b 1408 -ub 384
```

### With Q8 MTP Speculative Drafter (+47.6% Speedup)
```bash
GGML_VK_VISIBLE_DEVICES=0 GGML_VK_ALLOW_GRAPHICS_QUEUE=1 RADV_PERFTEST=nogttspill \
llama-server \
  -m gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  -md mtp-gemma-4-12B-it-Q8_0.gguf \
  -draft-n-max 3 \
  -ngl 99 -fa on --no-mmap -c 16384 -np 1 -ctk q8_0 -ctv q8_0 -b 1408 -ub 384
```

---

## 📊 Full Benchmark Data

### Mellum2 12B MoE Context Scaling Matrix (Single Vega 56 GPU, 8 GB VRAM)
| Context Allocation (`-c`) | Prompt Tokens | Prompt Processing (PP) | Text Generation (TG) | VRAM Occupancy & Allocation Behavior | SHA-256 Match |
|---|:---:|:---:|:---:|---|:---:|
| **16k (`16384`)** | 5 598 | 748.6 tok/s | 61.4 tok/s | ~90% VRAM Resident | `38e0b9de81...` |
| **32k (`32768`)** | 5 598 | 747.4 tok/s | 61.5 tok/s | ~91% VRAM Resident | `38e0b9de81...` |
| **64k (`65536`)** | 5 598 | 884.7 tok/s | 64.0 tok/s | ~91.5% VRAM Resident | `38e0b9de81...` |
| **128k (`131072`)** | 5 598 | **750.4 tok/s** | **61.1 tok/s** | **~92% VRAM Resident (7.51 GB)** | `38e0b9de81...` |
| **176k (`180224`)** | 5 598 | **751.3 tok/s** | **59.9 tok/s** | **~97% VRAM Resident** | `38e0b9de81...` |
| **224k (`229376`)** | 5 598 | **674.2 tok/s** | **14.9 tok/s** | **~99% VRAM Upper Limit** | `38e0b9de81...` |
| **256k (`262144`)** | 5 598 | 92.8 tok/s | 0.9 tok/s | 99.9% VRAM GTT Thrashing Cliff | `38e0b9de81...` |

### Gemma 4 12B Speculative Decoding Matrix
| Configuration | Draft `n_max` | Prefill (PP) | Decode (TG) | Draft Acceptance | Decode Speedup |
|---|:---:|:---:|:---:|:---:|:---:|
| **Solo Gemma 4 12B** | None | **233.3 tok/s** | **17.6 tok/s** | — | Baseline |
| **Gemma 4 12B + Q8 Drafter** | 2 | 230.5 tok/s | **25.7 tok/s** | **83.2%** | **+46.6%** |
| **Gemma 4 12B + Q8 Drafter** | **3** | **230.2 tok/s** | **25.9 tok/s** | **73.7%** | **+47.6%** |
| **Gemma 4 12B + Q8 Drafter** | 4 | 230.2 tok/s | **25.0 tok/s** | 65.7% | +42.6% |
| **Gemma 4 12B + Q8 Drafter** | 5 | 229.9 tok/s | 20.5 tok/s | 62.8% | +16.6% |

---

## 🌐 Multi-GPU Cluster Setup (7x Vega 56 for Bifrost Proxy)

For multi-GPU rigs (e.g. 7x Vega GPUs on mining risers), run 7 independent single-GPU `llama-server` instances so each card handles its own 128k context stream without PCIe bottlenecking:

```bash
# Starts 7 llama-server backends on ports 8001..8007 (GPUs 0..6)
./benchmarks/llm/launch-7gpu-mellum2-cluster.sh
```

Bifrost proxy can route requests directly to:
- `http://0.0.0.0:8001/v1` (GPU 0)
- `http://0.0.0.0:8002/v1` (GPU 1)
- ...
- `http://0.0.0.0:8007/v1` (GPU 6)

**Aggregate Performance**: **~440–470 TG tok/s** across 7 concurrent streams with **887 MiB free host RAM** on a 3.8 GB system.

---

## 📚 Deep-Dive Documentation
- [ANALYSIS.md](file:///home/orchestra/orca/projects/vega-bonsai/ANALYSIS.md): Comprehensive microarchitecture deep-dive, GCN occupancy traps, Vulkan kernel optimization work, and multi-GPU pipeline analysis.
- [RESULTS.md](file:///home/orchestra/orca/projects/vega-bonsai/RESULTS.md): Chronological experiment log and raw benchmark ledger.
- [patches/](file:///home/orchestra/orca/projects/vega-bonsai/patches): Custom Vulkan GCN kernel patches (`0001-vulkan-wide-ternary-mmv-gcn-knobs.patch`).
