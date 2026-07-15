"""Upload papers with multipart form data"""
import urllib.request, urllib.error, json, os, glob, sys, time
sys.stdout.reconfigure(encoding="utf-8")

base = "http://127.0.0.1:8766"
raw = r"C:\Users\lc202\.openclaw\workspace\local-kb-v2\data\raw_arxiv"
files = sorted(glob.glob(os.path.join(raw, "paper_*.txt")))
print(f"{len(files)} files")

ok = 0
for i, fp in enumerate(files):
    fn = os.path.basename(fp)
    with open(fp, "rb") as f:
        raw_data = f.read()
    
    # Build multipart manually
    boundary = f"----FormBoundary{i}"
    
    meta = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fn}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8")
    
    footer = (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="source_type"\r\n\r\n'
        f"学术论文\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    
    body = meta + raw_data + footer
    
    req = urllib.request.Request(f"{base}/upload", data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
            if d.get("status") == "ok":
                ok += 1
                if ok % 10 == 0:
                    print(f"{ok}/{len(files)}", end=" ", flush=True)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", "ignore")[:100]
        print(f"\nERR at {i}: HTTP {e.code} {err_body}")
        time.sleep(2)
    except Exception as e:
        print(f"\nERR at {i}: {e}")
        time.sleep(2)

print(f"\nOK: {ok}/{len(files)}")

# Final status
with urllib.request.urlopen(f"{base}/status", timeout=5) as r:
    print(json.loads(r.read()))
