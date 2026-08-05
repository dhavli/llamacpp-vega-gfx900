#!/usr/bin/env bash
# Qwen long-prompt core-clock A/B. Keeps HBM at state 3, verifies every core
# clock commit, and always restores cards 1-3 to 1590 MHz / 1200 mV.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/scb4cmx0h15sfbrapkjyx0r5jrzv8gpi-vega-runtime}
model=${MODEL:-/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
sclk=${SCLK:-1700}
voltage=${VOLTAGE:-1200}

set_sclk() {
    local mhz=$1 mv=$2 card
    for card in 1 2 3; do
        printf 's 7 %s %s\n' "${mhz}" "${mv}" \
            > "/sys/class/drm/card${card}/device/pp_od_clk_voltage"
        printf 'c\n' > "/sys/class/drm/card${card}/device/pp_od_clk_voltage"
        printf '7\n' > "/sys/class/drm/card${card}/device/pp_dpm_sclk"
        printf '3\n' > "/sys/class/drm/card${card}/device/pp_dpm_mclk"
    done
    sleep 3
    for card in 1 2 3; do
        sed -n '/OD_SCLK:/,/OD_MCLK:/p' \
            "/sys/class/drm/card${card}/device/pp_od_clk_voltage" \
            | grep -Eq "^7:[[:space:]]+${mhz}Mhz[[:space:]]+${mv}mV"
        grep -Eq "^7:[[:space:]]+${mhz}Mhz[[:space:]]+\\*" \
            "/sys/class/drm/card${card}/device/pp_dpm_sclk"
    done
}
trap 'set +e; set_sclk 1590 1200' EXIT

run_pp() {
    local label=$1
    GGML_VK_VISIBLE_DEVICES=0,1,2 timeout 1800 "${runtime}/bin/llama-bench" \
        -m "${model}" -ngl 99 -fa 1 --mmap 0 \
        -p 2048,8192 -n 0 -r 2 \
        2>&1 | tee "/tmp/core-pp-${label}.log"
}

set_sclk 1590 1200
run_pp baseline-1590
set_sclk "${sclk}" "${voltage}"
run_pp "${sclk}-${voltage}"
