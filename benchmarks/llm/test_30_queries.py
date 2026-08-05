#!/usr/bin/env python3
"""
30 Concurrent OpenAI API Queries Load Tester for 7-GPU Mellum2 Proxy (port 8000).
Fires 30 requests in parallel, measures latency, throughput, backend load distribution,
and total aggregate tokens generated per second.
"""
import urllib.request
import urllib.error
import json
import time
import concurrent.futures
import sys

PROXY_URL = "http://127.0.0.1:8000/v1/completions"
NUM_QUERIES = 30

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute the Fibonacci sequence efficiently.",
    "What are the key benefits of using Rust over C++ for system programming?",
    "Summarize the main principles of quantum mechanics.",
    "Draft a professional email requesting a project deadline extension.",
    "Explain how database indexes work under the hood using B-trees.",
    "What is the difference between synchronous and asynchronous I/O?",
    "Write a short essay on the impact of artificial intelligence on healthcare.",
    "Explain the concept of memory alignment and cache lines in modern CPUs.",
    "What are vector embeddings and how are they used in semantic search?"
]

def send_query(query_id):
    prompt_text = PROMPTS[query_id % len(PROMPTS)]
    payload = json.dumps({
        "prompt": f"User query #{query_id+1}: {prompt_text}",
        "max_tokens": 64,
        "temperature": 0.2
    }).encode("utf-8")

    req = urllib.request.Request(
        PROXY_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            t1 = time.time()
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = t1 - t0
            usage = data.get("usage", {})
            timings = data.get("timings", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            tg_tps = timings.get("predicted_per_second", 0.0)
            pp_tps = timings.get("prompt_per_second", 0.0)
            
            return {
                "id": query_id + 1,
                "status": resp.status,
                "elapsed": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tg_tps": tg_tps,
                "pp_tps": pp_tps,
                "error": None
            }
    except Exception as e:
        t1 = time.time()
        return {
            "id": query_id + 1,
            "status": 500,
            "elapsed": t1 - t0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tg_tps": 0.0,
            "pp_tps": 0.0,
            "error": str(e)
        }

def main():
    print(f"=== Starting 30 Concurrent Queries Load Test on Proxy {PROXY_URL} ===")
    print(f"Firing 30 parallel HTTP requests across the 7-GPU cluster...")
    
    start_time = time.time()
    results = []
    
    # Fire 30 concurrent requests using 30 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(send_query, i) for i in range(NUM_QUERIES)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"Query #{res['id']:02d} completed in {res['elapsed']:.2f}s | status: {res['status']} | completion tok: {res['completion_tokens']} | tg_tps: {res['tg_tps']:.1f}")

    total_wall_time = time.time() - start_time
    
    successful = [r for r in results if r['status'] == 200]
    failed = [r for r in results if r['status'] != 200]
    
    total_prompt_tok = sum(r['prompt_tokens'] for r in successful)
    total_completion_tok = sum(r['completion_tokens'] for r in successful)
    avg_latency = sum(r['elapsed'] for r in successful) / len(successful) if successful else 0
    min_latency = min(r['elapsed'] for r in successful) if successful else 0
    max_latency = max(r['elapsed'] for r in successful) if successful else 0
    
    aggregate_decode_tps = total_completion_tok / total_wall_time if total_wall_time > 0 else 0
    
    print("\n========================================================")
    print("               LOAD TEST RESULTS SUMMARY                ")
    print("========================================================")
    print(f"Total Requests:            {NUM_QUERIES}")
    print(f"Successful Requests:       {len(successful)} / {NUM_QUERIES} ({len(successful)/NUM_QUERIES*100:.1f}%)")
    print(f"Failed Requests:           {len(failed)}")
    print(f"Total Wall-Clock Time:     {total_wall_time:.2f} seconds")
    print(f"Total Completion Tokens:   {total_completion_tok} tokens")
    print(f"Total Prompt Tokens:       {total_prompt_tok} tokens")
    print(f"Average Request Latency:   {avg_latency:.2f} seconds")
    print(f"Min / Max Latency:         {min_latency:.2f}s / {max_latency:.2f}s")
    print(f"Aggregate Cluster Decode:  {aggregate_decode_tps:.2f} tokens/sec")
    print("========================================================\n")

if __name__ == "__main__":
    main()
