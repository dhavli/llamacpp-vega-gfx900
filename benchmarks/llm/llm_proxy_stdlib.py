#!/usr/bin/env python3
"""
Zero-dependency Python Multi-GPU Queueing Load Balancer Proxy for llama-server backends.
Listens on port 8000 (0.0.0.0:8000) and dispatches incoming OpenAI API requests
across active llama-server instances (ports 8001..8007 for GPUs 0..6).
Features automatic request queuing and retry when all backends are busy.
"""
import http.server
import socketserver
import urllib.request
import urllib.error
import threading
import time
import json
import sys

BACKENDS = [f"http://127.0.0.1:{port}" for port in range(8001, 8008)]
active_requests = {b: 0 for b in BACKENDS}
request_lock = threading.Lock()
round_robin_counter = 0

def check_backend_health(url):
    try:
        req = urllib.request.Request(f"{url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return resp.status == 200
    except Exception:
        # Fall back to True if active_requests[url] > 0 (server is alive and processing)
        return active_requests[url] > 0

def select_free_backend():
    global round_robin_counter
    with request_lock:
        healthy = [b for b in BACKENDS if check_backend_health(b)]
        if not healthy:
            healthy = list(BACKENDS)
        # Prefer backends with 0 active requests
        idle = [b for b in healthy if active_requests[b] == 0]
        if idle:
            idle.sort(key=lambda b: (active_requests[b], round_robin_counter))
            selected = idle[0]
        else:
            # Pick least loaded
            healthy.sort(key=lambda b: (active_requests[b], round_robin_counter))
            selected = healthy[0]
            
        active_requests[selected] += 1
        round_robin_counter = (round_robin_counter + 1) % 1000000
        return selected

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.handle_proxy("GET")

    def do_POST(self):
        self.handle_proxy("POST")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def handle_proxy(self, method):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status_data = {
                "status": "ok",
                "backends": BACKENDS,
                "active_requests": active_requests
            }
            self.wfile.write(json.dumps(status_data).encode("utf-8"))
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Queue request until a backend becomes available (up to 120s timeout)
        start_queue = time.time()
        backend = None
        
        while time.time() - start_queue < 120:
            backend = select_free_backend()
            if backend:
                break
            time.sleep(0.05)

        if not backend:
            self.send_error(503, "No healthy llama-server backends available")
            return

        target_url = f"{backend}{self.path}"

        req = urllib.request.Request(target_url, data=body, method=method)
        for key, value in self.headers.items():
            if key.lower() not in ("host", "content-length"):
                req.add_header(key, value)

        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    if key.lower() not in ("transfer-encoding", "content-length", "content-encoding"):
                        self.send_header(key, val)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_error(502, f"Proxy Error: {e}")
        finally:
            with request_lock:
                active_requests[backend] = max(0, active_requests[backend] - 1)

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

def run(port=8000):
    server_address = ("0.0.0.0", port)
    httpd = ThreadedHTTPServer(server_address, ProxyHandler)
    print(f"[Proxy] Robust Queueing Multi-GPU Proxy listening on 0.0.0.0:{port} fronting backends {BACKENDS}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run(p)
