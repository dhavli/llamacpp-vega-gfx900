#!/usr/bin/env bash
# Production-shaped Qwen four-card HBM A/B. Uses the real 5629-token prompt,
# verifies every OD commit, byte-compares greedy output, and restores 800 MHz.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/scb4cmx0h15sfbrapkjyx0r5jrzv8gpi-vega-runtime}
model=${MODEL:-/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
prompt=${PROMPT:-/root/bonsai/prompt8k.txt}
mclk=${MCLK:-900}

set_mclk() {
    local mhz=$1 card
    for card in 1 2 3 4; do
        printf 'm 3 %s 950\n' "${mhz}" \
            > "/sys/class/drm/card${card}/device/pp_od_clk_voltage"
        printf 'c\n' > "/sys/class/drm/card${card}/device/pp_od_clk_voltage"
        printf '7\n' > "/sys/class/drm/card${card}/device/pp_dpm_sclk"
        printf '3\n' > "/sys/class/drm/card${card}/device/pp_dpm_mclk"
    done
    sleep 3
    for card in 1 2 3 4; do
        sed -n '/OD_MCLK:/,/OD_RANGE:/p' \
            "/sys/class/drm/card${card}/device/pp_od_clk_voltage" \
            | grep -Eq "^3:[[:space:]]+${mhz}Mhz[[:space:]]+950mV"
        grep -Eq "^3:[[:space:]]+${mhz}Mhz[[:space:]]+\\*" \
            "/sys/class/drm/card${card}/device/pp_dpm_mclk"
    done
}
trap 'set +e; set_mclk 800' EXIT

run_completion() {
    local label=$1
    GGML_VK_VISIBLE_DEVICES=0,1,2,3 timeout 1200 \
        "${runtime}/bin/llama-completion" \
        -m "${model}" -ngl 99 -fa on --no-mmap -c 8192 -n 128 \
        --temp 0 --ignore-eos -no-cnv -f "${prompt}" \
        > "/tmp/four-card-hbm-${label}.txt" \
        2> "/tmp/four-card-hbm-${label}.log"
    grep -E 'prompt eval time|eval time' "/tmp/four-card-hbm-${label}.log"
}

set_mclk 800
run_completion baseline-800
set_mclk "${mclk}"
run_completion "${mclk}"

if cmp -s /tmp/four-card-hbm-baseline-800.txt \
        "/tmp/four-card-hbm-${mclk}.txt"; then
    echo "four-card-hbm-${mclk}: IDENTICAL"
else
    echo "four-card-hbm-${mclk}: DIFFERS"
    exit 1
fi
