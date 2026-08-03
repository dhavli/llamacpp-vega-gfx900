#!/usr/bin/env bash
# Cold versus resident-slot exact-prefix reuse on the frozen two-GPU Gemma pod.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/4wybfk3s46cdjm2gfnzm3zhxkzgp1qsv-vega-runtime}
model=${MODEL:-/root/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}
prompt=${PROMPT:-/root/bonsai/prompt8k.txt}
devices=${DEVICES:-0,1}
port=${PORT:-8093}
label=${LABEL:-gemma4-prefix-cache}
n_batch=${BATCH:-1408}
backend_sampling=${BACKEND_SAMPLING:-0}
output_reserve=${OUTPUT_RESERVE:-128}
server_pid=

cleanup() {
    if [[ -n ${server_pid} ]]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_server() {
    local phase=$1
    local extra_args=()
    if [[ ${backend_sampling} == 1 ]]; then
        extra_args+=(--backend-sampling)
    fi
    env GGML_VK_VISIBLE_DEVICES="${devices}" GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
        RADV_PERFTEST=nogttspill LLAMA_SERVER_OUTPUT_RESERVE="${output_reserve}" \
        "${runtime}/bin/llama-server" -m "${model}" -ngl 99 -fa on --no-mmap \
        -c 131072 -np 1 -ctk q8_0 -ctv q8_0 -b "${n_batch}" -ub 384 \
        --cache-ram 0 --ctx-checkpoints 0 --host 127.0.0.1 --port "${port}" \
        "${extra_args[@]}" \
        > "/tmp/${label}-${phase}.server.log" 2>&1 &
    server_pid=$!

    local ready=0
    for _ in $(seq 1 180); do
        if curl --silent --max-time 2 "http://127.0.0.1:${port}/health" | grep -q ok; then
            ready=1
            break
        fi
        kill -0 "${server_pid}" 2>/dev/null || break
        sleep 3
    done
    if [[ ${ready} -ne 1 ]]; then
        tail -n 60 "/tmp/${label}-${phase}.server.log"
        exit 1
    fi
}

request() {
    local name=$1
    local body=$2
    curl --fail --silent --show-error --max-time 900 \
        -o "/tmp/${label}-${name}.json" -w '%{time_total}\n' \
        -X POST "http://127.0.0.1:${port}/v1/completions" \
        -H 'Content-Type: application/json' --data-binary "@${body}" \
        > "/tmp/${label}-${name}.wall"
}

prefix_body="/tmp/${label}-prefix.request.json"
full_body="/tmp/${label}-full.request.json"
jq -nc --rawfile p "${prompt}" \
    '{prompt:$p,n_predict:1,temperature:0,ignore_eos:true,cache_prompt:true}' \
    > "${prefix_body}"
jq -nc --rawfile p "${prompt}" \
    '{prompt:($p+"\nSummarize the central claim in one sentence."),n_predict:128,
      temperature:0,ignore_eos:true,cache_prompt:true}' > "${full_body}"

start_server cold
request cold-full "${full_body}"
cleanup
server_pid=
sleep 3

start_server warm
request warm-prefix "${prefix_body}"
request warm-full "${full_body}"

for name in cold-full warm-prefix warm-full; do
    jq -nc --arg name "${name}" \
        --argjson wall "$(cat "/tmp/${label}-${name}.wall")" \
        --slurpfile response "/tmp/${label}-${name}.json" \
        '{name:$name,wall_seconds:$wall,prompt_tokens:$response[0].usage.prompt_tokens,
          generated_tokens:$response[0].usage.completion_tokens,
          pp_tps:$response[0].timings.prompt_per_second,
          tg_tps:$response[0].timings.predicted_per_second,
          cached_tokens:($response[0].timings.cache_n // null)}'
done
for name in cold-full warm-full; do
    jq -r .content "/tmp/${label}-${name}.json" | sha256sum
done
