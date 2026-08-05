#!/usr/bin/env bash
# Deterministic task-quality matrix for configurable MoE expert counts.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/hycp2s33y3mpv5cslr6ghly05rmm8kqy-vega-runtime}
model=${MODEL:-/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
port=${PORT:-8089}
expert_counts=${EXPERT_COUNTS:-8 7 6}
architecture=${ARCHITECTURE:-qwen35moe}
devices=${DEVICES:-0,1,2,3}
n_ctx=${CONTEXT:-4096}
n_batch=${BATCH:-2048}
n_ubatch=${UBATCH:-768}
tensor_split=${TENSOR_SPLIT-0.85,1.05,1.05,1.05}
per_queue_mutex=${PER_QUEUE_MUTEX:-1}
server_pid=

cleanup() {
    if [[ -n ${server_pid} ]]; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

prompts=(
    'A warehouse starts with 240 bolts. It ships 3/8 of them, then packs the remainder into boxes of 12. Return exactly: shipped=<integer>; full_boxes=<integer>; leftover=<integer>'
    'List every prime from 90 through 110 inclusive and their sum. Return exactly: primes=<comma-separated integers>; sum=<integer>'
    'In Python, what does this print? x=[1,2,3]; y=x; y += [4]; z=x[:]; z.append(5); print(x, z). Return only the two printed list values.'
    'Review this Python function: def append_item(item, items=[]): items.append(item); return items. State the bug in one sentence, then give a corrected function. Keep the whole answer under 80 words.'
    'Facts: Every vel is a nor. No nor is a zim. Some paks are zims. Which conclusion must be true? A) No vel is a zim. B) No pak is a vel. C) Some nors are paks. D) Some zims are vels. Return only the letter.'
    'Transform the sequence 7, -2, 7, 4, -2, 0: remove duplicates, sort ascending, then square each value. Return only a JSON array of numbers.'
    'A service has current value 10. The operation PUT /counter with body {"value":15} is successfully applied, but the response is lost, so the identical request is retried. Assume PUT replaces the value. Return exactly: final_value=<integer>; applications=<integer>; idempotent=<yes|no>'
    'A log contains: user=ana action=login code=200; user=bo action=upload code=503; user=ana action=logout code=200. Return a minified JSON array containing only records whose code is not 200, with keys in this order: user,action,code.'
)

request_tasks() {
    local label=$1
    local output="/tmp/${label}.tasks.jsonl"
    : > "${output}"

    for i in "${!prompts[@]}"; do
        local task=$((i + 1))
        local body response
        body=$(jq -nc --arg p "${prompts[$i]}" \
            '{messages:[{role:"user",content:$p}],max_tokens:160,temperature:0,
              chat_template_kwargs:{enable_thinking:false}}')
        response=$(curl --fail --silent --show-error --max-time 900 \
            -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "${body}")
        jq -e '.choices[0].message.content != null and .choices[0].finish_reason != null' \
            <<< "${response}" >/dev/null
        jq -nc --argjson task "${task}" --arg prompt "${prompts[$i]}" \
            --argjson response "${response}" \
            '{task:$task,prompt:$prompt,text:$response.choices[0].message.content,
              finish_reason:$response.choices[0].finish_reason,
              prompt_tps:$response.timings.prompt_per_second,
              generation_tps:$response.timings.predicted_per_second}' >> "${output}"
        echo "${label}: task ${task} complete"
    done
}

run_config() {
    local label=$1
    shift

    echo "##### ${label} args=[$*]"
    local server_env=(GGML_VK_VISIBLE_DEVICES="${devices}" GGML_VK_ALLOW_GRAPHICS_QUEUE=1)
    local split_args=()
    if [[ ${per_queue_mutex} == 1 ]]; then
        server_env+=(GGML_VK_PER_QUEUE_MUTEX=1)
    fi
    if [[ -n ${tensor_split} ]]; then
        split_args+=(-ts "${tensor_split}")
    fi
    env "${server_env[@]}" \
        "${runtime}/bin/llama-server" -m "${model}" -ngl 99 -fa on --no-mmap \
        -c "${n_ctx}" -np 1 -b "${n_batch}" -ub "${n_ubatch}" "${split_args[@]}" \
        --reasoning-budget 0 --reasoning-format none \
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

for expert_count in ${expert_counts}; do
    if [[ ${expert_count} == 8 ]]; then
        run_config e8-tasks-final
    else
        run_config "e${expert_count}-tasks-final" \
            --override-kv "${architecture}.expert_used_count=int:${expert_count}"
    fi
done
echo '### EXPERTTASKDONE'
