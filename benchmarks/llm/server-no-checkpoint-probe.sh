#!/usr/bin/env bash
# Probe llama-server on the VRAM-tight 3-card pod with recurrent checkpoints
# and the server prompt cache disabled. Defaults create ~62.8 MiB checkpoints
# and ~129 MiB prompt-cache entries for this hybrid model.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/0rlr67bjyf76wfzf1mmzgq73pk22fk5x-vega-runtime}
model=${MODEL:-/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
port=${PORT:-8089}
label=server-no-checkpoints

GGML_VK_VISIBLE_DEVICES=0,1,2 \
    "${runtime}/bin/llama-server" -m "${model}" -ngl 99 -fa on --no-mmap \
    -c 4096 -np 1 --ctx-checkpoints 0 --cache-ram 0 \
    --reasoning-budget 0 --reasoning-format none \
    --host 127.0.0.1 --port "${port}" > "/tmp/${label}.log" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 180); do
    if curl --silent --max-time 2 "http://127.0.0.1:${port}/health" | grep -q ok; then
        ready=1
        break
    fi
    kill -0 "${server_pid}" 2>/dev/null || break
    sleep 5
done
[[ ${ready} -eq 1 ]]

body=$(jq -nc '{messages:[{role:"user",content:"Answer concisely and show your reasoning. If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?"}],max_tokens:256,temperature:0,chat_template_kwargs:{enable_thinking:false}}')
curl --fail --silent --show-error --max-time 900 \
    -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "${body}" > "/tmp/${label}.json"
jq '{text:.choices[0].message.content,prompt_tps:.timings.prompt_per_second,generation_tps:.timings.predicted_per_second}' \
    "/tmp/${label}.json"
echo '### SERVERNOCHECKPOINTDONE'
