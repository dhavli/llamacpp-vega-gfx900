#!/usr/bin/env bash
# Production-shape llama-server probe: four 128k slots on four cards, with
# a configurable number of simultaneous 5.6k-token prompts and deterministic generations.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/hycp2s33y3mpv5cslr6ghly05rmm8kqy-vega-runtime}
model=${MODEL:-/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
prompt_file=${PROMPT:-/root/bonsai/prompt8k.txt}
port=${PORT:-8089}
label=${LABEL:-server-4x128k}
graphics_queue=${GRAPHICS_QUEUE:-1}
per_queue_mutex=${PER_QUEUE_MUTEX:-0}
requests=${REQUESTS:-1}
n_batch=${BATCH:-2048}
n_ubatch=${UBATCH:-512}
tensor_split=${TENSOR_SPLIT-}
use_mmap=${USE_MMAP:-0}
expert_count=${EXPERT_COUNT:-8}

while pgrep -f '[c]oldcard-finder' >/dev/null; do
    echo 'coldcard-finder owns the GPUs; waiting'
    sleep 60
done

server_env=(RADV_PERFTEST=nogttspill)
if [[ ${graphics_queue} == 1 ]]; then
    server_env+=(GGML_VK_ALLOW_GRAPHICS_QUEUE=1)
fi
if [[ ${per_queue_mutex} == 1 ]]; then
    server_env+=(GGML_VK_PER_QUEUE_MUTEX=1)
fi

mmap_args=()
if [[ ${use_mmap} == 0 ]]; then
    mmap_args+=(--no-mmap)
fi

split_args=()
if [[ -n ${tensor_split} ]]; then
    split_args+=(-ts "${tensor_split}")
fi

expert_args=()
if [[ ${expert_count} != 8 ]]; then
    expert_args+=(--override-kv "qwen35moe.expert_used_count=int:${expert_count}")
fi

env "${server_env[@]}" \
GGML_VK_VISIBLE_DEVICES=0,1,2,3 LLAMA_SERVER_FULL_OUTPUT_RESERVE=1 \
    "${runtime}/bin/llama-server" -m "${model}" -ngl 99 -fa on "${mmap_args[@]}" \
    -ctk q8_0 -ctv q8_0 -c $((4 * 131072)) -np 4 \
    -b "${n_batch}" -ub "${n_ubatch}" "${split_args[@]}" \
    "${expert_args[@]}" \
    --cache-ram 0 --ctx-checkpoints 0 \
    --host 127.0.0.1 --port "${port}" > "/tmp/${label}.server.log" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 240); do
    if curl --silent --max-time 2 "http://127.0.0.1:${port}/health" | grep -q ok; then
        ready=1
        break
    fi
    kill -0 "${server_pid}" 2>/dev/null || break
    sleep 5
done
[[ ${ready} -eq 1 ]]

body=$(jq -nc --rawfile p "${prompt_file}" \
    '{prompt:$p,n_predict:128,temperature:0,ignore_eos:true,cache_prompt:false}')
request_pids=()
for slot in $(seq 1 "${requests}"); do
    curl --fail --silent --show-error --max-time 1800 \
        -X POST "http://127.0.0.1:${port}/v1/completions" \
        -H 'Content-Type: application/json' -d "${body}" \
        > "/tmp/${label}.slot${slot}.json" &
    request_pids+=("$!")
done
for request_pid in "${request_pids[@]}"; do
    wait "${request_pid}"
done

for slot in $(seq 1 "${requests}"); do
    jq -c --argjson slot "${slot}" \
        '{slot:$slot,prompt_tokens:.usage.prompt_tokens,generated_tokens:.usage.completion_tokens,
          pp_tps:.timings.prompt_per_second,tg_tps:.timings.predicted_per_second}' \
        "/tmp/${label}.slot${slot}.json"
done
echo "### SERVER4X128KDONE requests=${requests}"
