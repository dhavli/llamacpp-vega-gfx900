#!/usr/bin/env bash
# Low-host-RAM quality A/B for Qwen3.6's normal top-8 routing versus top-6.
set -u

runtime=${RUNTIME:-/nix/store/0rlr67bjyf76wfzf1mmzgq73pk22fk5x-vega-runtime}
model=${MODEL:-/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
prompt=${PROMPT:-/root/bonsai/prompt8k.txt}

run_ppl() {
    local label=$1
    shift

    echo "##### ${label} args=[$*]"
    GGML_VK_VISIBLE_DEVICES=0,1,2 \
        timeout 1800 "${runtime}/bin/llama-perplexity" \
        -m "${model}" -ngl 99 -fa on --no-mmap -c 2048 -b 128 \
        -f "${prompt}" "$@" > "/tmp/${label}.pplout" 2>&1
    local rc=$?
    tail -n 8 "/tmp/${label}.pplout"
    echo "${label}: exit=${rc}"
    return "${rc}"
}

run_ppl p5-e8
run_ppl p5-e6 --override-kv qwen35moe.expert_used_count=int:6
echo "### P5DONE"
