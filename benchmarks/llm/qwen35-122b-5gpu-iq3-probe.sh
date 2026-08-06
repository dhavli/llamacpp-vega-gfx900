#!/usr/bin/env bash
# Qwen 3.5 122B A10B IQ3 probe for 5-GPU AMD host (5x RX 6900 XT / 80GB VRAM)
set -euo pipefail

runtime=${RUNTIME:-/nix/store/f57pns4a5iyzf54zvbr8sgr92y9s57nf-vega-runtime}
model=${MODEL:-/root/models/Qwen3.5-122B-A10B-UD-IQ3_XXS.gguf}
prompt=${PROMPT:-/root/bonsai/prompt8k.txt}
devices=${DEVICES:-0,1,2,3,4}
port=${PORT:-8098}
label=${LABEL:-qwen35-122b-iq3xxs-5gpu}
n_batch=${BATCH:-2048}
n_ubatch=${UBATCH:-512}
n_ctx=${CTX:-131072}
n_predict=${N_PREDICT:-128}

server_args=(
    -m "${model}" -ngl 999 -fa on --no-mmap
    -c "${n_ctx}" -np 1 -ctk q4_0 -ctv q4_0
    -b "${n_batch}" -ub "${n_ubatch}"
    --cache-ram 0 --ctx-checkpoints 0
    --host 127.0.0.1 --port "${port}"
)

echo "Starting llama-server for ${label} on GPUs ${devices}..."
env GGML_VK_VISIBLE_DEVICES="${devices}" \
    GGML_VK_ALLOW_GRAPHICS_QUEUE=1 RADV_PERFTEST=nogttspill \
    LLAMA_SERVER_OUTPUT_RESERVE=128 \
    "${runtime}/bin/llama-server" "${server_args[@]}" \
    > "/tmp/${label}.server.log" 2>&1 &
server_pid=$!
trap 'kill -9 "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 300); do
    response=$(curl --silent --show-error --max-time 2 \
        "http://127.0.0.1:${port}/health" 2>/dev/null || true)
    if grep -q ok <<< "${response}"; then
        ready=1
        break
    fi
    kill -0 "${server_pid}" 2>/dev/null || break
    sleep 2
done

if [[ ${ready} -ne 1 ]]; then
    echo "=== Server failed to start ===" >&2
    tail -n 60 "/tmp/${label}.server.log" >&2
    exit 1
fi

request_body="/tmp/${label}.request.json"
jq -nc --rawfile p "${prompt}" \
    --argjson np "${n_predict}" \
    '{prompt:$p,n_predict:$np,temperature:0,ignore_eos:true,cache_prompt:false}' \
    > "${request_body}"

echo "Sending completion request..."
curl --fail --silent --show-error --max-time 900 \
    -X POST "http://127.0.0.1:${port}/v1/completions" \
    -H 'Content-Type: application/json' --data-binary "@${request_body}" \
    > "/tmp/${label}.json"

output_hash=$(jq -r .content "/tmp/${label}.json" | sha256sum | awk '{print $1}')

jq -c --arg lbl "${label}" --arg hash "${output_hash}" \
    '{label:$lbl, prompt_tokens:.usage.prompt_tokens, generated_tokens:.usage.completion_tokens, pp_tps:.timings.prompt_per_second, tg_tps:.timings.predicted_per_second, sha256:$hash}' \
    "/tmp/${label}.json"
