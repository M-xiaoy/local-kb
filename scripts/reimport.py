"""重新导入 v2 知识库 — 触发因果密度切分"""
import os, sys, json, glob, time, urllib.request, urllib.error
sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8766"
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_arxiv")

def wait_for_server(max_wait=30):
    for i in range(max_wait):
        try:
            r = urllib.request.Request(f"{BASE}/status")
            with urllib.request.urlopen(r, timeout=3) as resp:
                return json.loads(resp.read())
        except:
            time.sleep(1)
    return None

def upload_file(filepath):
    """上传论文，source_type=学术论文 触发因果密度切分"""
    filename = os.path.basename(filepath)
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    with open(filepath, "rb") as f:
        file_data = f.read()
    
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode() + file_data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="source_type"\r\n\r\n'
        f"学术论文\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="auto_rebuild"\r\n\r\n'
        f"false\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    for attempt in range(2):
        try:
            req = urllib.request.Request(f"{BASE}/upload", data=body)
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "ignore")[:200]
            time.sleep(2)
            return {"status": "error", "message": err}
        except Exception as e:
            time.sleep(2)
    return {"status": "error", "message": "timeout"}


print("=" * 60)
print("v2 知识库 — 因果密度切分重导入")
print("=" * 60)

# 1. 确认服务运行
print("\n[1] 检查服务器...")
status = wait_for_server()
if status is None:
    print("  ! 服务器未运行，请先启动")
    sys.exit(1)
print(f"  当前: {status['total_spheres']} 球体")

# 2. 清空数据（删除 sphere_store + faiss 索引）
print("\n[2] 清空旧数据...")
import shutil
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
for subdir in ["spheres", "index", "connections", "uploads", "wal"]:
    target = os.path.join(data_dir, subdir)
    if os.path.exists(target):
        shutil.rmtree(target)
        print(f"  已删除: {subdir}/")
os.makedirs(os.path.join(data_dir, "spheres"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "index"), exist_ok=True)

# 3. 重启服务（清空内存状态）
print("\n[3] 重启服务以清空内存状态...")
# 向 /rebuild 发空请求，触发重新构建（用空数据）
import urllib.parse
try:
    req = urllib.request.Request(f"{BASE}/rebuild", data=b"{}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=300) as r:
        print(f"  重建结果: {json.loads(r.read())}")
except Exception as e:
    print(f"  重建中: {e}")
    time.sleep(3)

# 4. 批量导入论文
print(f"\n[4] 批量导入论文（因果密度切分）...")
files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
print(f"  待导入: {len(files)} 篇")

success = 0
span_count = 0
for i, fp in enumerate(files):
    fn = os.path.basename(fp)
    print(f"  [{i+1}/{len(files)}] {fn[:50]}...", end="", flush=True)
    r = upload_file(fp)
    if r.get("status") == "ok":
        spans = r.get("total_spans", 1)
        span_count += spans
        success += 1
        print(f" ✓ ({spans} blocks)", flush=True)
    else:
        print(f" ✗ {r.get('message', '?')}", flush=True)
    time.sleep(0.8)

print(f"\n  导入成功: {success}/{len(files)}")
print(f"  总球体块: {span_count}")

# 5. 重建 FAISS 索引
print(f"\n[5] 重建 FAISS 索引...")
try:
    req = urllib.request.Request(f"{BASE}/rebuild", data=b"{}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=300) as r:
        print(f"  结果: {json.loads(r.read())}")
except Exception as e:
    print(f"  失败: {e}")

# 6. 重建连接（树突 + 轴突 + 方向）
print(f"\n[6] 重建连接（含轴突方向）...")
try:
    req = urllib.request.Request(f"{BASE}/rebuild-connections", data=b"{}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=300) as r:
        result = json.loads(r.read())
        print(f"  总连接: {result['total_connections']}")
        print(f"  轴突: {result.get('axon_edges', 0)}")
        print(f"  平均 degree: {result['avg_degree']}")
except Exception as e:
    print(f"  失败: {e}")

# 7. 最终状态
print(f"\n[7] 最终状态...")
with urllib.request.urlopen(f"{BASE}/status", timeout=5) as r:
    status = json.loads(r.read())
    print(f"  球体: {status['total_spheres']}")
    print(f"  向量: {status['faiss_vectors']}")
    print(f"  簇: {len(status.get('fields', []))}")
