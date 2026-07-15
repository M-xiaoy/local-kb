"""Upload paper_ files to v2, one at a time"""
import urllib.request, urllib.error, json, os, sys, glob, time

sys.stdout.reconfigure(encoding="utf-8")
base = "http://127.0.0.1:8766"
raw = r"C:\Users\lc202\.openclaw\workspace\local-kb-v2\data\raw_arxiv"
files = sorted(glob.glob(os.path.join(raw, "paper_*.txt")))

print(f"Uploading {len(files)} papers...")
ok = 0
for i, fp in enumerate(files):
    fn = os.path.basename(fp)
    with open(fp, "rb") as f:
        raw_data = f.read()
    
    boundary = "----B" + str(i)
    meta = b"--" + boundary.encode() + b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\"" + fn.encode() + b"\"\r\nContent-Type: text/plain\r\n\r\n"
    footer = b"\r\n--" + boundary.encode() + b"\r\nContent-Disposition: form-data; name=\"source_type\"\r\n\r\n" + "学术论文".encode("utf-8") + b"\r\n--" + boundary.encode() + b"--\r\n"
    body = meta + raw_data + footer
    
    req = urllib.request.Request(base + "/upload", data=body)
    req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
            ok += 1
            if ok % 10 == 0:
                print(f"{ok}/{len(files)}", flush=True)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")[:80]
        print(f"  ERR {fn[:30]}: HTTP {e.code} {msg}", flush=True)
    except Exception as e:
        print(f"  ERR {fn[:30]}: {type(e).__name__}", flush=True)
    time.sleep(0.5)

print(f"\nOK: {ok}/{len(files)}")
r = json.loads(urllib.request.urlopen(base + "/status", timeout=5).read())
print(f"Total spheres: {r['total_spheres']}")
