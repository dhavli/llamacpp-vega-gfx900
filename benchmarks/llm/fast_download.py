import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

if len(sys.argv) < 3:
    print("Usage: python3 fast_download.py <url> <output_path> [num_threads]")
    sys.exit(1)

url = sys.argv[1]
output_path = sys.argv[2]
num_threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16

req = urllib.request.Request(url, method='HEAD')
# Follow redirects if HEAD returns 302/301
class NoRedirect(urllib.request.HTTPRedirectHandler):
    pass

opener = urllib.request.build_opener()
resp = opener.open(req)
final_url = resp.geturl()
total_size = int(resp.headers.get('Content-Length'))
print(f"Target size: {total_size} bytes. Downloading via {num_threads} threads from {final_url}")

with open(output_path, 'wb') as f:
    f.truncate(total_size)

chunk_size = total_size // num_threads

def download_chunk(part_idx):
    start = part_idx * chunk_size
    end = total_size - 1 if part_idx == num_threads - 1 else (part_idx + 1) * chunk_size - 1
    req = urllib.request.Request(final_url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req) as r:
        with open(output_path, 'r+b') as f:
            f.seek(start)
            while True:
                buf = r.read(1024 * 1024)
                if not buf:
                    break
                f.write(buf)

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(download_chunk, i) for i in range(num_threads)]
    for i, f in enumerate(futures):
        f.result()
        print(f"Chunk {i+1}/{num_threads} completed.")

print(f"Successfully downloaded {output_path} ({os.path.getsize(output_path)} bytes)")
