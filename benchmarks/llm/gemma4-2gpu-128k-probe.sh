#!/usr/bin/env bash
# Gemma 4 26B-A4B text-only single-pod probe: two Vega cards, one 128K slot.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/4wybfk3s46cdjm2gfnzm3zhxkzgp1qsv-vega-runtime}
model=${MODEL:-/root/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}
draft=${DRAFT:-/root/models/mtp-gemma-4-26B-A4B-it.gguf}
prompt=${PROMPT:-/root/bonsai/prompt8k.txt}
devices=${DEVICES:-0,1}
port=${PORT:-8093}
label=${LABEL:-gemma4-2gpu-128k}
n_batch=${BATCH:-1408}
n_ubatch=${UBATCH:-384}
use_mtp=${USE_MTP:-0}
backend_sampling=${BACKEND_SAMPLING:-0}
tensor_split=${TENSOR_SPLIT-}
expert_count=${EXPERT_COUNT:-8}
output_reserve=${OUTPUT_RESERVE:-128}

server_args=(
    -m "${model}" -ngl 99 -fa on --no-mmap
    -c 131072 -np 1 -ctk q8_0 -ctv q8_0
    -b "${n_batch}" -ub "${n_ubatch}"
    --cache-ram 0 --ctx-checkpoints 0
    --host 127.0.0.1 --port "${port}"
)
if [[ ${use_mtp} == 1 ]]; then
    server_args+=(--spec-draft-model "${draft}" --spec-type draft-mtp
        --spec-draft-n-max 4 --spec-draft-ngl 99)
fi
if [[ ${backend_sampling} == 1 ]]; then
    server_args+=(--backend-sampling)
fi
if [[ -n ${tensor_split} ]]; then
    server_args+=(-ts "${tensor_split}")
fi
if [[ ${expert_count} != 8 ]]; then
    server_args+=(--override-kv "gemma4.expert_used_count=int:${expert_count}")
fi

env GGML_VK_VISIBLE_DEVICES="${devices}" \
    GGML_VK_ALLOW_GRAPHICS_QUEUE=1 RADV_PERFTEST=nogttspill \
    LLAMA_SERVER_OUTPUT_RESERVE="${output_reserve}" \
    "${runtime}/bin/llama-server" "${server_args[@]}" \
    > "/tmp/${label}.server.log" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 180); do
    response=$(curl --silent --show-error --max-time 2 \
        "http://127.0.0.1:${port}/health" 2>/dev/null || true)
    if grep -q ok <<< "${response}"; then
        ready=1
        break
    fi
    kill -0 "${server_pid}" 2>/dev/null || break
    sleep 3
done
if [[ ${ready} -ne 1 ]]; then
    tail -n 60 "/tmp/${label}.server.log"
    exit 1
fi

request_body="/tmp/${label}.request.json"
jq -nc --rawfile p "${prompt}" \
    '{prompt:$p,n_predict:128,temperature:0,ignore_eos:true,cache_prompt:false}' \
    > "${request_body}"
curl --fail --silent --show-error --max-time 900 \
    -X POST "http://127.0.0.1:${port}/v1/completions" \
    -H 'Content-Type: application/json' --data-binary "@${request_body}" \
    > "/tmp/${label}.json"
jq -c '{prompt_tokens:.usage.prompt_tokens,generated_tokens:.usage.completion_tokens,
        pp_tps:.timings.prompt_per_second,tg_tps:.timings.predicted_per_second}' \
    "/tmp/${label}.json"
jq -r .content "/tmp/${label}.json" | sha256sum
if [[ ${use_mtp} == 1 ]]; then
    grep -Ei 'draft acceptance|statistics.*draft-mtp' "/tmp/${label}.server.log" || true
fi
