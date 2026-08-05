#!/usr/bin/env python3
"""
Lightweight OpenAI-compatible HTTP Load Balancer Proxy for multi-card llama-server backends.
Listens on port 8000 and dispatches requests across active llama-server instances (ports 8001..8007).
Uses round-robin / least-connections dispatch with automatic health check filtering.
"""
import asyncio
import http.client
import json
import logging
import sys
from urllib.parse import urlparse
from aiohttp import web, ClientSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BACKENDS = [f"http://127.0.0.1:{port}" for port in range(8001, 8008)]
active_requests = {b: 0 for b in BACKENDS}
backend_idx = 0
lock = asyncio.Lock()

async def get_next_backend():
    global backend_idx
    async with lock:
        # Select healthy backend with least active requests
        healthy_backends = []
        async with ClientSession() as session:
            for b in BACKENDS:
                try:
                    async with session.get(f"{b}/health", timeout=1.0) as resp:
                        if resp.status == 200:
                            healthy_backends.append(b)
                except Exception:
                    pass
        
        if not healthy_backends:
            return None
        
        # Sort by active requests, then round-robin
        healthy_backends.sort(key=lambda b: active_requests[b])
        selected = healthy_backends[0]
        active_requests[selected] += 1
        return selected

async def proxy_handler(request):
    backend = await get_next_backend()
    if not backend:
        return web.json_response({"error": "No healthy llama-server backends available"}, status=503)
    
    path = request.path_qs
    method = request.method
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ('host', 'content-length')}
    body = await request.read()
    
    logging.info(f"Proxying {method} {path} -> {backend} (Active on backend: {active_requests[backend]})")
    
    try:
        async with ClientSession() as session:
            async with session.request(method, f"{backend}{path}", headers=headers, data=body, timeout=900) as resp:
                resp_body = await resp.read()
                resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in ('transfer-encoding', 'content-length', 'content-encoding')}
                return web.Response(body=resp_body, status=resp.status, headers=resp_headers)
    except Exception as e:
        logging.error(f"Error forwarding to {backend}: {e}")
        return web.json_response({"error": str(e)}, status=502)
    finally:
        async with lock:
            active_requests[backend] = max(0, active_requests[backend] - 1)

async def health_check(request):
    return web.json_response({"status": "ok", "backends": BACKENDS, "active_requests": active_requests})

app = web.Application()
app.router.add_get('/health', health_check)
app.router.add_route('*', '/{tail:.*}', proxy_handler)

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    logging.info(f"Starting LLM Proxy on 0.0.0.0:{port} fronting backends {BACKENDS}")
    web.run_app(app, host='0.0.0.0', port=port)
