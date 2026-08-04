#!/usr/bin/env bash
# Sweep Mellum2 context window with Q8 KV quantization ONLY to fill up to ~99% VRAM on single Vega 56 card
set -euo pipefail

MODEL="/root/models/Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf"
HARNESS="/root/bonsai/mellum2-max-ctx-probe.sh"
OUTFILE="/root/bonsai/mellum2-maxctx-results.jsonl"

rm -f "${OUTFILE}"
echo "=== Starting Mellum2 12B Max Context Sweep (Q8 KV ONLY, Single Vega 56 GPU) ==="

# Fine-grained Q8 KV context size sweep above 128k to find 99% VRAM limit
# 128k (131072), 144k (147456), 160k (163840), 176k (180224), 192k (196608), 208k (212992), 224k (229376), 256k (262144)
for ctx in 131072 147456 163840 180224 196608 212992 229376 262144; do
    echo "--------------------------------------------------------"
    echo "Testing CTX=${ctx} (Q8 KV ONLY)..."
    echo "--------------------------------------------------------"
    pkill -9 llama-server || true
    sleep 2
    if CTX="${ctx}" KV_TYPE="q8_0" PORT=8096 "${HARNESS}" | tee -a "${OUTFILE}"; then
        echo "SUCCESS: CTX=${ctx} (Q8 KV) passed and executed."
    else
        echo "MAX VRAM CEILING REACHED for Q8 KV at CTX=${ctx}"
        break
    fi
    sleep 2
done

echo "=== All Mellum2 Max Context Q8 KV Sweep passes finished ==="
cat "${OUTFILE}"
