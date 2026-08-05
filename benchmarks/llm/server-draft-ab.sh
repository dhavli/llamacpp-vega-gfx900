#!/usr/bin/env bash
# Run on the Vega rig. Three target GPUs stay fixed; the small draft gets GPU 3.
# Draft provenance: unsloth/Qwen3.5-0.8B-GGUF, Qwen3.5-0.8B-Q4_K_M.gguf,
# sha256 bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517.
# It was removed from the disk-constrained rig after the falsification.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/scb4cmx0h15sfbrapkjyx0r5jrzv8gpi-vega-runtime}
model=${MODEL:-/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
draft=${DRAFT:-/root/models/Qwen3.5-0.8B-Q4_K_M.gguf}
prompt=${PROMPT:-/root/bonsai/prompt8k.txt}
port=${PORT:-8090}
server_pid=

stop_server() {
    if [[ -n ${server_pid} ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}"
        wait "${server_pid}" 2>/dev/null || true
    fi
    server_pid=
}
trap stop_server EXIT

run_case() {
    local label=$1
    shift
    local log="/tmp/draft-${label}.server.log"
    local json="/tmp/draft-${label}.json"

    GGML_VK_VISIBLE_DEVICES=0,1,2,3 \
    LLAMA_SERVER_FULL_OUTPUT_RESERVE=1 \
    "${runtime}/bin/llama-server" \
        -m "${model}" -ngl 99 -fa on --no-mmap \
        --device Vulkan0,Vulkan1,Vulkan2 \
        -c 8192 -np 1 --cache-ram 0 --ctx-checkpoints 0 \
        --host 127.0.0.1 --port "${port}" "$@" > "${log}" 2>&1 &
    server_pid=$!

    for _ in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
            break
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            tail -n 80 "${log}" >&2
            return 1
        fi
        sleep 1
    done
    curl -sf "http://127.0.0.1:${port}/health" >/dev/null

    jq -n --rawfile prompt "${prompt}" \
        '{prompt:$prompt,n_predict:256,temperature:0,ignore_eos:true,cache_prompt:false}' \
        | curl -sf "http://127.0.0.1:${port}/completion" \
            -H 'Content-Type: application/json' --data-binary @- > "${json}"

    jq '{timings, tokens_predicted, content}' "${json}"
    stop_server
}

[[ -f ${draft} ]] || {
    echo "missing draft model: ${draft}" >&2
    exit 1
}

run_case control
run_case nmax3 \
    --spec-type draft-simple --spec-draft-model "${draft}" \
    --spec-draft-ngl 99 --spec-draft-device Vulkan3 --spec-draft-n-max 3
run_case nmax5 \
    --spec-type draft-simple --spec-draft-model "${draft}" \
    --spec-draft-ngl 99 --spec-draft-device Vulkan3 --spec-draft-n-max 5
