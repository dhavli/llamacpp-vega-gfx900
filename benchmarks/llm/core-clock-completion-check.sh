#!/usr/bin/env bash
# Separate-process deterministic Vega core-clock check. Generates a fresh stock
# clock oracle and always restores cards 1-3 to 1590 MHz / 1200 mV.
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
        # Committing an OD table resets both DPM selections on this driver.
        printf '7\n' > "/sys/class/drm/card${card}/device/pp_dpm_sclk"
        printf '3\n' > "/sys/class/drm/card${card}/device/pp_dpm_mclk"
    done
    sleep 3
    for card in 1 2 3; do
        if ! sed -n '/OD_SCLK:/,/OD_MCLK:/p' \
                "/sys/class/drm/card${card}/device/pp_od_clk_voltage" \
                | grep -Eq "^7:[[:space:]]+${mhz}Mhz[[:space:]]+${mv}mV"; then
            echo "card${card}: requested ${mhz} MHz / ${mv} mV was not applied" >&2
            return 1
        fi
        if ! grep -Eq "^7:[[:space:]]+${mhz}Mhz[[:space:]]+\\*" \
                "/sys/class/drm/card${card}/device/pp_dpm_sclk"; then
            echo "card${card}: core DPM state 7 is not active" >&2
            return 1
        fi
    done
}
trap 'set +e; set_sclk 1590 1200' EXIT

run_completion() {
    local label=$1
    GGML_VK_VISIBLE_DEVICES=0,1,2 timeout 1200 "${runtime}/bin/llama-completion" \
        -m "${model}" -ngl 99 -fa on --no-mmap -c 4096 -n 128 \
        --temp 0 --ignore-eos -no-cnv \
        -p 'If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?' \
        > "/tmp/core-completion-${label}.txt" 2> "/tmp/core-completion-${label}.log"
    grep -E 'prompt eval time|eval time' "/tmp/core-completion-${label}.log"
}

set_sclk 1590 1200
run_completion baseline-1590
set_sclk "${sclk}" "${voltage}"
run_completion "${sclk}-${voltage}"

if cmp -s /tmp/core-completion-baseline-1590.txt \
        "/tmp/core-completion-${sclk}-${voltage}.txt"; then
    echo "core-completion-${sclk}-${voltage}: IDENTICAL"
else
    echo "core-completion-${sclk}-${voltage}: DIFFERS"
    exit 1
fi
