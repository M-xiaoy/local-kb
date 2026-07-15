"""批量导入论文到 v2 知识库"""
import os, sys, json, time, glob, urllib.request, urllib.error

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8766"
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_arxiv")


def check_server():
    """等待服务器就绪"""
    for i in range(30):
        try:
            req = urllib.request.Request(f"{BASE}/status")
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())
                print(f"服务器就绪: {data.get('active', 0)} 活跃球体")
                return True
        except Exception as e:
            if i % 5 == 0:
                print(f"等待服务器启动... ({i}s)")
            time.sleep(1)
    print("服务器未就绪")
    return False


def upload_file(filepath, source_type="学术论文", max_retries=3):
    """上传单个文件到知识库"""
    filename = os.path.basename(filepath)
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

    with open(filepath, "rb") as f:
        file_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + file_data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="source_type"\r\n\r\n'
        f"{source_type}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="auto_rebuild"\r\n\r\n'
        f"false\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                f"{BASE}/upload",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                result = json.loads(r.read())
                return result
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", "ignore")[:200]
            print(f"  HTTP {e.code}: {err_body}")
            time.sleep(2)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)

    return {"status": "error", "message": "Max retries exceeded"}


def main():
    if not check_server():
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
    print(f"\n准备导入 {len(files)} 篇论文")

    success = 0
    failed = 0

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(files)}] {filename}", end="")

        result = upload_file(filepath)

        if result.get("status") == "ok":
            sphere_id = result.get("sphere_id", "?")
            print(f" → ✓ ({sphere_id})")
            success += 1
        else:
            print(f" → ✗ {result.get('message', 'unknown error')}")
            failed += 1

        # 每篇间隔 0.5s
        time.sleep(0.5)

    print(f"\n=== 导入完成 ===")
    print(f"成功: {success}, 失败: {failed}")

    # 触发重建连接和校准
    if success > 0:
        print("\n触发连接重建...")
        try:
            req = urllib.request.Request(
                f"{BASE}/rebuild-connections",
                data=b"{}",
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                print(json.loads(r.read()))
        except Exception as e:
            print(f"重建连接失败: {e}")

        print("\n触发校准...")
        try:
            req = urllib.request.Request(
                f"{BASE}/calibrate",
                data=b"{}",
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                print(json.loads(r.read()))
        except Exception as e:
            print(f"校准失败: {e}")


if __name__ == "__main__":
    main()
