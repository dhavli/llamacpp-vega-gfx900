#!/usr/bin/env bash
# Benchmark harness for Gemma 4 12B Q4_K_XL on single Vega 56 GPU
set -euo pipefail

MODEL="/root/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
DRAFT="/root/models/mtp-gemma-4-12B-it-Q8_0.gguf"
PROMPT="/root/bonsai/prompt8k.txt"
HARNESS="/root/bonsai/gemma4-12b-1gpu-probe.sh"
OUTFILE="/root/bonsai/gemma4-12b-results.jsonl"

echo "=== Starting Gemma 4 12B single-GPU evaluation harness ==="
echo "Model: ${MODEL}"
echo "Drafter: ${DRAFT}"
echo "Output: ${OUTFILE}"

# Wait for downloads if still in progress
while true; do
    if [[ -f "${MODEL}" && -f "${DRAFT}" ]]; then
        sz_m=$(stat -c%s "${MODEL}" 2>/dev/null || echo 0)
        sz_d=$(stat -c%s "${DRAFT}" 2>/dev/null || echo 0)
        # Expected size: MODEL > 6.5 GB (6500000000), DRAFT > 450 MB (450000000)
        if [[ ${sz_m} -ge 6500000000 && ${sz_d} -ge 450000000 ]]; then
            echo "Models present and full size: Model=${sz_m} bytes, Drafter=${sz_d} bytes."
            break
        fi
    fi
    echo "Waiting for model downloads to complete... (Model: $(stat -c%s "${MODEL}" 2>/dev/null || echo 0), Drafter: $(stat -c%s "${DRAFT}" 2>/dev/null || echo 0))"
    sleep 10
done

rm -f "${OUTFILE}"

# Pass 1: Solo (without drafter)
echo "=== Pass 1: Solo Gemma 4 12B (No Drafter) ==="
LABEL="gemma4-12b-solo" USE_MTP=0 CTX=16384 N_PREDICT=128 DEVICES=0 PORT=8095 "${HARNESS}" | tee -a "${OUTFILE}"
sleep 2

# Pass 1b: Solo with longer generation (256 tokens)
echo "=== Pass 1b: Solo Gemma 4 12B (No Drafter, 256 tokens) ==="
LABEL="gemma4-12b-solo-256tok" USE_MTP=0 CTX=16384 N_PREDICT=256 DEVICES=0 PORT=8095 "${HARNESS}" | tee -a "${OUTFILE}"
sleep 2

# Pass 2a: With Q8 Drafter (n_max=2)
echo "=== Pass 2a: Gemma 4 12B + Q8 Drafter (n_max=2) ==="
LABEL="gemma4-12b-q8draft-n2" USE_MTP=1 DRAFT_N_MAX=2 CTX=16384 N_PREDICT=128 DEVICES=0 PORT=8095 "${HARNESS}" | tee -a "${OUTFILE}"
sleep 2

# Pass 2b: With Q8 Drafter (n_max=3)
echo "=== Pass 2b: Gemma 4 12B + Q8 Drafter (n_max=3) ==="
LABEL="gemma4-12b-q8draft-n3" USE_MTP=1 DRAFT_N_MAX=3 CTX=16384 N_PREDICT=128 DEVICES=0 PORT=8095 "${HARNESS}" | tee -a "${OUTFILE}"
sleep 2

# Pass 2c: With Q8 Drafter (n_max=4)
echo "=== Pass 2c: Gemma 4 12B + Q8 Drafter (n_max=4) ==="
LABEL="gemma4-12b-q8draft-n4" USE_MTP=1 DRAFT_N_MAX=4 CTX=16384 N_PREDICT=128 DEVICES=0 PORT=8095 "${HARNESS}" | tee -a "${OUTFILE}"
sleep 2

# Pass 2d: With Q8 Drafter (n_max=5)
echo "=== Pass 2d: Gemma 4 12B + Q8 Drafter (n_max=5) ==="
LABEL="gemma4-12b-q8draft-n5" USE_MTP=1 DRAFT_N_MAX=5 CTX=16384 N_PREDICT=128 DEVICES=0 PORT=8095 "${HARNESS}" | tee -a "${OUTFILE}"
sleep 2

echo "=== All Gemma 4 12B single-GPU evaluation passes finished ==="
cat "${OUTFILE}"
