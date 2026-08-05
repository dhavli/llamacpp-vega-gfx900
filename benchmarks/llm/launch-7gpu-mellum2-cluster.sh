#!/usr/bin/env bash
# Robust 7-GPU Mellum2 12B MoE Cluster Launcher (ports 8001..8007 for GPUs 0..6)
# + Zero-Dependency Multi-GPU Load Balancer Proxy on port 8000.
set -euo pipefail

runtime=${RUNTIME:-/nix/store/f57pns4a5iyzf54zvbr8sgr92y9s57nf-vega-runtime}
model=${MODEL:-/root/models/Mellum2-12B-A2.5B-Thinking-MXFP4_MOE.gguf}
ctx=${CTX:-131072}
proxy_port=${PROXY_PORT:-8000}

echo "=== Initializing 7-GPU Mellum2 12B MoE Cluster (128k context, Q8 KV) ==="

pkill -9 llama-server || true
pkill -f llm_proxy || true
sleep 3

# Launch servers sequentially with a short delay to prevent CPU contention during Vulkan init
for gpu in $(seq 0 6); do
    port=$((8001 + gpu))
    label="mellum2-gpu${gpu}"
    echo "Launching GPU ${gpu} backend on port ${port}..."
    server_args=(
        -m "${model}" -ngl 99 -fa on --no-mmap
        -c "${ctx}" -np 1 -ctk q8_0 -ctv q8_0
        -b 1408 -ub 384 --cache-ram 0 --ctx-checkpoints 0
        --host 127.0.0.1 --port "${port}"
    )
    env GGML_VK_VISIBLE_DEVICES="${gpu}" \
        GGML_VK_ALLOW_GRAPHICS_QUEUE=1 RADV_PERFTEST=nogttspill \
        LLAMA_SERVER_OUTPUT_RESERVE=128 \
        "${runtime}/bin/llama-server" "${server_args[@]}" \
        > "/tmp/${label}.server.log" 2>&1 &
    sleep 1.5
done

echo "Waiting for all 7 llama-server backends to reach READY status..."
for _ in $(seq 1 60); do
    ready_count=0
    for gpu in $(seq 0 6); do
        port=$((8001 + gpu))
        if curl --silent --max-time 1 "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; then
            ready_count=$((ready_count + 1))
        fi
    done
    echo "Ready backends: ${ready_count}/7"
    if [[ ${ready_count} -eq 7 ]]; then
        break
    fi
    sleep 3
done

echo "Launching Zero-Dependency Load Balancer Proxy on 0.0.0.0:${proxy_port}..."
nohup python3 /root/bonsai/llm_proxy_stdlib.py "${proxy_port}" > /tmp/llm_proxy.log 2>&1 &
sleep 2

echo "=== Final Cluster Status ==="
curl --silent "http://127.0.0.1:${proxy_port}/health" || true
echo ""
echo "Host RAM Usage:"
free -h
