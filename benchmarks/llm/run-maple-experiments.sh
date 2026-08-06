#!/usr/bin/env bash
# Comprehensive 2-GPU Benchmark Suite for Maple Preview Q4_K_M
set -euo pipefail

MODEL="/root/models/maple-q4_k_m.gguf"
HARNESS="/root/bonsai/maple-2gpu-probe.sh"
OUTFILE="/root/bonsai/maple-results.jsonl"

rm -f "${OUTFILE}"
echo "=== Starting Maple Preview Q4_K_M 2-GPU Benchmark Suite ==="

# Pass 1: 16k context, Q8 KV, 2 GPUs (0,1)
echo "--------------------------------------------------------"
echo "Pass 1: 16k context (Q8 KV, 2 GPUs)"
echo "--------------------------------------------------------"
pkill -9 llama-server || true; sleep 2
LABEL="maple-q4km-ctx16384-q8" CTX=16384 KV_TYPE="q8_0" PORT=8096 "${HARNESS}" | tee -a "${OUTFILE}"

# Pass 2: 32k context, Q8 KV, 2 GPUs (0,1)
echo "--------------------------------------------------------"
echo "Pass 2: 32k context (Q8 KV, 2 GPUs)"
echo "--------------------------------------------------------"
pkill -9 llama-server || true; sleep 2
LABEL="maple-q4km-ctx32768-q8" CTX=32768 KV_TYPE="q8_0" PORT=8096 "${HARNESS}" | tee -a "${OUTFILE}"

# Pass 3: 64k context, Q8 KV, 2 GPUs (0,1)
echo "--------------------------------------------------------"
echo "Pass 3: 64k context (Q8 KV, 2 GPUs)"
echo "--------------------------------------------------------"
pkill -9 llama-server || true; sleep 2
LABEL="maple-q4km-ctx65536-q8" CTX=65536 KV_TYPE="q8_0" PORT=8096 "${HARNESS}" | tee -a "${OUTFILE}"

# Pass 4: 128k context, Q8 KV, 2 GPUs (0,1)
echo "--------------------------------------------------------"
echo "Pass 4: 128k context (Q8 KV, 2 GPUs)"
echo "--------------------------------------------------------"
pkill -9 llama-server || true; sleep 2
if LABEL="maple-q4km-ctx131072-q8" CTX=131072 KV_TYPE="q8_0" PORT=8096 "${HARNESS}" | tee -a "${OUTFILE}"; then
    echo "128k Q8 KV passed!"
else
    echo "128k Q8 KV hit VRAM limit, trying 128k Q4 KV..."
    pkill -9 llama-server || true; sleep 2
    LABEL="maple-q4km-ctx131072-q4" CTX=131072 KV_TYPE="q4_0" PORT=8096 "${HARNESS}" | tee -a "${OUTFILE}"
fi

# Pass 5: Batch/Ubatch Tuning (b=2048, ub=768) at 32k context
echo "--------------------------------------------------------"
echo "Pass 5: High-batch tuning (b=2048, ub=768) at 32k context"
echo "--------------------------------------------------------"
pkill -9 llama-server || true; sleep 2
LABEL="maple-q4km-b2048-ub768-ctx32768" CTX=32768 KV_TYPE="q8_0" BATCH=2048 UBATCH=768 PORT=8096 "${HARNESS}" | tee -a "${OUTFILE}"

echo "=== All Maple Preview Q4_K_M 2-GPU Benchmark Passes Completed ==="
cat "${OUTFILE}"
