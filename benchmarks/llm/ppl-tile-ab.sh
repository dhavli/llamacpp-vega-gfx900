#!/usr/bin/env bash
# Low-host-RAM perplexity A/B for the Qwen3.6 MoE TILE_M candidate.
# The rig's 3.8 GiB host OOM-kills -b 512 when its 508 MiB logits buffer is
# materialized.  -b 128 limits that buffer to about 127 MiB; --no-mmap avoids
# retaining a 21 GiB model mapping while the logits are evaluated.
set -u

runtime=${RUNTIME:-/nix/store/0rlr67bjyf76wfzf1mmzgq73pk22fk5x-vega-runtime}
model=${MODEL:-/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
prompt=${PROMPT:-/root/bonsai/prompt8k.txt}
tile=${TILE_M:-256,128,64,64,32,64,2,4,4,1,64}

run_ppl() {
    local label=$1
    local tile_value=${2:-}

    echo "##### ${label} tile=${tile_value:-default}"
    GGML_VK_VISIBLE_DEVICES=0,1,2 \
        env ${tile_value:+GGML_VK_TILE_M=${tile_value}} \
        timeout 1800 "${runtime}/bin/llama-perplexity" \
        -m "${model}" -ngl 99 -fa on --no-mmap -c 2048 -b 128 \
        -f "${prompt}" > "/tmp/${label}.pplout" 2>&1
    local rc=$?
    tail -n 8 "/tmp/${label}.pplout"
    echo "${label}: exit=${rc}"
    return "${rc}"
}

run_ppl p4-ctrl
run_ppl p4-tile "${tile}"
echo "### P4DONE"
