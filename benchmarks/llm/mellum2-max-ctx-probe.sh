#!/usr/bin/env bash
# Max context probe for Mellum2 12B MoE on single Vega 56 GPU (8 GiB VRAM)
set -euo pipefail

runtime=${RUNTIME:-/nix/store/f57pns4a5iyzf54zvbr8sgr92y9s57nf-vega-runtime}
model=${MODEL:-/root/models/Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf}
prompt=${PROMPT:-/root/bonsai/prompt8k.txt}
devices=${DEVICES:-0}
port=${PORT:-8096}
n_batch=${BATCH:-1408}
n_ubatch=${UBATCH:-384}
kv_type=${KV_TYPE:-q8_0}
output_reserve=${OUTPUT_RESERVE:-128}

echo "=== Testing Mellum2 max context limit: CTX=${CTX}, KV=${kv_type} on GPU ${devices} ==="

label="mellum2-maxctx-${CTX}-${kv_type}"
server_args=(
    -m "${model}" -ngl 99 -fa on --no-mmap
    -c "${CTX}" -np 1 -ctk "${kv_type}" -ctv "${kv_type}"
    -b "${n_batch}" -ub "${n_ubatch}"
    --cache-ram 0 --ctx-checkpoints 0
    --host 127.0.0.1 --port "${port}"
)

env GGML_VK_VISIBLE_DEVICES="${devices}" \
    GGML_VK_ALLOW_GRAPHICS_QUEUE=1 RADV_PERFTEST=nogttspill \
    LLAMA_SERVER_OUTPUT_RESERVE="${output_reserve}" \
    "${runtime}/bin/llama-server" "${server_args[@]}" \
    > "/tmp/${label}.server.log" 2>&1 &
server_pid=$!
trap 'kill -9 "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 120); do
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
    echo "=== Server FAILED to start for CTX=${CTX} KV=${kv_type} ==="
    tail -n 35 "/tmp/${label}.server.log"
    exit 1
fi

echo "=== Server READY for CTX=${CTX} KV=${kv_type}. Extracting VRAM usage metrics... ==="
grep -Ei 'vram|free|buffer|offloaded|allocated' "/tmp/${label}.server.log" | tail -n 20 || true

request_body="/tmp/${label}.request.json"
jq -nc --rawfile p "${prompt}" \
    '{prompt:$p,n_predict:128,temperature:0,ignore_eos:true,cache_prompt:false}' \
    > "${request_body}"

curl --fail --silent --show-error --max-time 900 \
    -X POST "http://127.0.0.1:${port}/v1/completions" \
    -H 'Content-Type: application/json' --data-binary "@${request_body}" \
    > "/tmp/${label}.json"

output_hash=$(jq -r .content "/tmp/${label}.json" | sha256sum | awk '{print $1}')

jq -c --arg lbl "${label}" --arg hash "${output_hash}" --argjson ctx "${CTX}" --arg kv "${kv_type}" \
    '{label:$lbl, ctx:$ctx, kv:$kv, prompt_tokens:.usage.prompt_tokens, generated_tokens:.usage.completion_tokens, pp_tps:.timings.prompt_per_second, tg_tps:.timings.predicted_per_second, sha256:$hash}' \
    "/tmp/${label}.json"
