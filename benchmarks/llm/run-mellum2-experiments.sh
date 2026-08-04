#!/usr/bin/env bash
# Mellum2 12B MoE single GPU context scaling evaluation runner script
set -euo pipefail

MODEL="/root/models/Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf"
HARNESS="/root/bonsai/mellum2-1gpu-probe.sh"
OUTFILE="/root/bonsai/mellum2-results.jsonl"

echo "=== Starting Mellum2 12B single-GPU context scaling evaluation ==="
echo "Model: ${MODEL}"
echo "Output: ${OUTFILE}"

while true; do
    if [[ -f "${MODEL}" ]]; then
        sz=$(stat -c%s "${MODEL}" 2>/dev/null || echo 0)
        # Expected size: ~7,029,536,832 bytes
        if [[ ${sz} -ge 6800000000 ]]; then
            echo "Mellum2 model present and full size: ${sz} bytes."
            break
        fi
    fi
    echo "Waiting for Mellum2 download to complete... (Current size: $(stat -c%s "${MODEL}" 2>/dev/null || echo 0))"
    sleep 10
done

rm -f "${OUTFILE}"

# Context targets: 128k, 64k, 32k, 16k
for ctx in 131072 65536 32768 16384; do
    lbl="mellum2-12b-ctx${ctx}"
    echo "=========================================="
    echo "Testing Mellum2 12B with ctx=${ctx} (Q8 KV)..."
    echo "=========================================="
    if LABEL="${lbl}" CTX="${ctx}" DEVICES=0 PORT=8096 "${HARNESS}" | tee -a "${OUTFILE}"; then
        echo "Successfully completed test for ctx=${ctx}"
    else
        echo "FAILED to run with ctx=${ctx} (OOM or VRAM bound)"
    fi
    sleep 3
done

echo "=== All Mellum2 12B context scaling passes finished ==="
cat "${OUTFILE}"
