#!/usr/bin/env bash
# Separate-process deterministic HBM clock check. Generates a fresh 800 MHz
# llama-completion oracle and always restores cards to 800 MHz.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/scb4cmx0h15sfbrapkjyx0r5jrzv8gpi-vega-runtime}
model=${MODEL:-/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
mclk=${MCLK:-900}

set_mclk() {
    local mhz=$1 card
    for card in 1 2 3; do
        # Vega's OD parser requires the memory voltage even when it is unchanged.
        printf 'm 3 %s 950\n' "${mhz}" > "/sys/class/drm/card${card}/device/pp_od_clk_voltage"
        printf 'c\n' > "/sys/class/drm/card${card}/device/pp_od_clk_voltage"
        # Committing an OD table resets the DPM selections on this driver.
        printf '7\n' > "/sys/class/drm/card${card}/device/pp_dpm_sclk"
        printf '3\n' > "/sys/class/drm/card${card}/device/pp_dpm_mclk"
    done
    sleep 3
    for card in 1 2 3; do
        if ! sed -n '/OD_MCLK:/,/OD_RANGE:/p' \
                "/sys/class/drm/card${card}/device/pp_od_clk_voltage" \
                | grep -Eq "^3:[[:space:]]+${mhz}Mhz"; then
            echo "card${card}: requested ${mhz} MHz was not applied" >&2
            return 1
        fi
    done
}
trap 'set +e; set_mclk 800' EXIT

run_completion() {
    local label=$1
    GGML_VK_VISIBLE_DEVICES=0,1,2 timeout 1200 "${runtime}/bin/llama-completion" \
        -m "${model}" -ngl 99 -fa on --no-mmap -c 4096 -n 128 \
        --temp 0 --ignore-eos -no-cnv \
        -p 'If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?' \
        > "/tmp/hbm-completion-${label}.txt" 2> "/tmp/hbm-completion-${label}.log"
    grep -E 'prompt eval time|eval time' "/tmp/hbm-completion-${label}.log"
}

set_mclk 800
run_completion baseline-800
set_mclk "${mclk}"
run_completion "${mclk}"

if cmp -s /tmp/hbm-completion-baseline-800.txt "/tmp/hbm-completion-${mclk}.txt"; then
    echo "hbm-completion-${mclk}: IDENTICAL"
else
    echo "hbm-completion-${mclk}: DIFFERS"
    exit 1
fi
