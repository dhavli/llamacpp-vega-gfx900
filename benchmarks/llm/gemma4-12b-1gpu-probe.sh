#!/usr/bin/env bash
# Gemma 4 12B single-pod probe: single Vega card (GPU 0), with and without Q8 MTP drafter.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/f57pns4a5iyzf54zvbr8sgr92y9s57nf-vega-runtime}
model=${MODEL:-/root/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf}
draft=${DRAFT:-/root/models/mtp-gemma-4-12B-it-Q8_0.gguf}
prompt=${PROMPT:-/root/bonsai/prompt8k.txt}
devices=${DEVICES:-0}
port=${PORT:-8095}
label=${LABEL:-gemma4-12b-1gpu}
n_batch=${BATCH:-1408}
n_ubatch=${UBATCH:-384}
n_ctx=${CTX:-16384}
n_predict=${N_PREDICT:-128}
use_mtp=${USE_MTP:-0}
draft_n_max=${DRAFT_N_MAX:-4}
backend_sampling=${BACKEND_SAMPLING:-0}
output_reserve=${OUTPUT_RESERVE:-128}

server_args=(
    -m "${model}" -ngl 99 -fa on --no-mmap
    -c "${n_ctx}" -np 1 -ctk q8_0 -ctv q8_0
    -b "${n_batch}" -ub "${n_ubatch}"
    --cache-ram 0 --ctx-checkpoints 0
    --host 127.0.0.1 --port "${port}"
)

if [[ ${use_mtp} == 1 ]]; then
    server_args+=(--spec-draft-model "${draft}" --spec-type draft-mtp
        --spec-draft-n-max "${draft_n_max}" --spec-draft-ngl 99)
fi

if [[ ${backend_sampling} == 1 ]]; then
    server_args+=(--backend-sampling)
fi

env GGML_VK_VISIBLE_DEVICES="${devices}" \
    GGML_VK_ALLOW_GRAPHICS_QUEUE=1 RADV_PERFTEST=nogttspill \
    LLAMA_SERVER_OUTPUT_RESERVE="${output_reserve}" \
    "${runtime}/bin/llama-server" "${server_args[@]}" \
    > "/tmp/${label}.server.log" 2>&1 &
server_pid=$!
trap 'kill -9 "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 180); do
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

curl --fail --silent --show-error --max-time 900 \
    -X POST "http://127.0.0.1:${port}/v1/completions" \
    -H 'Content-Type: application/json' --data-binary "@${request_body}" \
    > "/tmp/${label}.json"

output_hash=$(jq -r .content "/tmp/${label}.json" | sha256sum | awk '{print $1}')

jq -c --arg lbl "${label}" --arg hash "${output_hash}" --argjson mtp "${use_mtp}" --argjson nmax "${draft_n_max}" \
    '{label:$lbl, mtp:$mtp, draft_n_max:$nmax, prompt_tokens:.usage.prompt_tokens, generated_tokens:.usage.completion_tokens, pp_tps:.timings.prompt_per_second, tg_tps:.timings.predicted_per_second, sha256:$hash}' \
    "/tmp/${label}.json"

if [[ ${use_mtp} == 1 ]]; then
    grep -Ei 'draft acceptance|statistics.*draft-mtp' "/tmp/${label}.server.log" || true
fi
