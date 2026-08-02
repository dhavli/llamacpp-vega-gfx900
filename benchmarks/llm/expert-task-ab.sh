#!/usr/bin/env bash
# Deterministic task-quality A/B for Qwen3.6 top-8 versus overridden top-6 routing.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/0rlr67bjyf76wfzf1mmzgq73pk22fk5x-vega-runtime}
model=${MODEL:-/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
port=${PORT:-8089}
server_pid=

cleanup() {
    if [[ -n ${server_pid} ]]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

prompts=(
    'Answer concisely and show your reasoning. If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?'
    'Review this Python function, identify the bug, and provide a corrected implementation: def append_item(item, items=[]): items.append(item); return items'
    'List every prime number from 90 through 120 inclusive, then give their sum. Show enough arithmetic to make the answer checkable.'
    'Explain whether HTTP PUT is idempotent and distinguish idempotence from safety. Give one concrete retry example. Keep the answer under 180 words.'
)

request_tasks() {
    local label=$1
    local output="/tmp/${label}.tasks.jsonl"
    : > "${output}"

    for i in "${!prompts[@]}"; do
        local task=$((i + 1))
        local body response
        body=$(jq -nc --arg p "${prompts[$i]}" \
            '{messages:[{role:"user",content:$p}],max_tokens:256,temperature:0,
              chat_template_kwargs:{enable_thinking:false}}')
        response=$(curl --fail --silent --show-error --max-time 900 \
            -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "${body}")
        jq -nc --argjson task "${task}" --arg prompt "${prompts[$i]}" \
            --argjson response "${response}" \
            '{task:$task,prompt:$prompt,text:$response.choices[0].message.content,
              prompt_tps:$response.timings.prompt_per_second,
              generation_tps:$response.timings.predicted_per_second}' >> "${output}"
        echo "${label}: task ${task} complete"
    done
}

run_config() {
    local label=$1
    shift

    echo "##### ${label} args=[$*]"
    GGML_VK_VISIBLE_DEVICES=0,1,2 \
        "${runtime}/bin/llama-server" -m "${model}" -ngl 99 -fa on --no-mmap \
        -c 4096 -np 1 --reasoning-budget 0 --reasoning-format none \
        --host 127.0.0.1 --port "${port}" "$@" \
        > "/tmp/${label}.server.log" 2>&1 &
    server_pid=$!

    local ready=0
    for _ in $(seq 1 180); do
        if curl --silent --max-time 2 "http://127.0.0.1:${port}/health" | grep -q ok; then
            ready=1
            break
        fi
        kill -0 "${server_pid}" 2>/dev/null || break
        sleep 5
    done
    if [[ ${ready} -ne 1 ]]; then
        echo "${label}: server failed to become healthy"
        tail -n 20 "/tmp/${label}.server.log"
        return 1
    fi

    request_tasks "${label}"
    cleanup
    server_pid=
    sleep 3
}

run_config e8-tasks
run_config e6-tasks --override-kv qwen35moe.expert_used_count=int:6
echo '### EXPERTTASKDONE'
