"""
usage: python kb_import_github.py [--rebuild]
批量从 GitHub 克隆教程仓库 → 提取 .md → 灌入 local-kb API.

前置条件:
  - local-kb API 运行在 http://127.0.0.1:8766
  - pip install httpx

每个仓库灌完后自动保存，进程中断不丢数据。
"""

import httpx, os, sys, subprocess, time, json
from pathlib import Path

API = "http://127.0.0.1:8766"
CACHE = Path(__file__).parent / ".repo_cache"

REPOS = [
    "https://github.com/datawhalechina/diy-llm.git",
    "https://github.com/datawhalechina/thorough-pytorch.git",
    "https://github.com/datawhalechina/happy-llm.git",
    "https://github.com/datawhalechina/leedl-tutorial.git",
    "https://github.com/PKUFlyingPig/cs-self-learning.git",
    "https://github.com/RiazML/math-for-llms.git",
]
EXCLUDE = {".git", "node_modules", "__pycache__", ".github",
           "images", "img", "assets", "fonts", "venv", ".venv", "dist", "build"}


def check_api():
    try:
        r = httpx.get(f"{API}/status", timeout=5)
        d = r.json()
        print(f"API OK: {d['active_spheres']} spheres, {d['faiss_vectors']} vectors")
        return True
    except Exception as e:
        print(f"API not reachable: {e}")
        print("Start the API first:")
        print("  cd local-kb && python -m uvicorn api.main:app --host 0.0.0.0 --port 8766 --workers 1")
        return False


def clone(url):
    name = url.split("/")[-1].replace(".git", "")
    target = CACHE / name
    if target.exists():
        print(f"  [SKIP] {name} already cloned")
        return name, target
    print(f"  [CLONE] {name} ...", end=" ", flush=True)
    r = subprocess.run(["git", "clone", "--depth", "1", url, str(target)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode == 0:
        print("OK")
    else:
        print(f"FAIL: {r.stderr[:120]}")
        return None, None
    return name, target


def find_md(repo_dir):
    files = []
    for root, dirs, _ in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE]
        for f in Path(root).glob("*.md"):
            if f.stat().st_size >= 500:
                files.append(f)
    return sorted(files)


def upload_file(md_path):
    with open(md_path, "rb") as fp:
        try:
            r = httpx.post(
                f"{API}/upload",
                files={"file": (md_path.name, fp, "text/markdown")},
                data={"source_type": "技术笔记", "auto_rebuild": "false"},
                timeout=180,
            )
            if r.status_code == 200:
                return r.json().get("new_spheres", 0)
            else:
                return f"ERR{r.status_code}"
        except Exception as e:
            return str(type(e).__name__)


def import_repo(name, repo_dir):
    mds = find_md(repo_dir)
    if not mds:
        print(f"  No .md files found")
        return

    print(f"  {len(mds)} .md files")
    total_new = 0
    t0 = time.time()
    for i, md in enumerate(mds):
        result = upload_file(md)
        if isinstance(result, int):
            total_new += result
        rel = str(md.relative_to(repo_dir))
        print(f"    [{i+1}/{len(mds)}] {rel[:55]:55s} {result}")
        if (i + 1) % 20 == 0:
            print(f"    → +{total_new} new, {time.time()-t0:.0f}s elapsed")

    print(f"  Done: +{total_new} new from {len(mds)} files ({time.time()-t0:.0f}s)")


def rebuild():
    print("\n--- Rebuilding ---")
    try:
        r = httpx.post(f"{API}/rebuild", timeout=600)
        print(f"  {r.status_code}: {r.text.strip()[:200]}")
    except Exception as e:
        print(f"  FAIL: {e}")
    r = httpx.get(f"{API}/status", timeout=10)
    d = r.json()
    print(f"Final: {d['active_spheres']} spheres")


if __name__ == "__main__":
    if not check_api():
        sys.exit(1)

    os.makedirs(CACHE, exist_ok=True)

    for url in REPOS:
        name, target = clone(url)
        if name is None:
            continue
        print()
        import_repo(name, target)

    if "--rebuild" in sys.argv:
        rebuild()
    else:
        print("\nDone. Run with --rebuild to rebuild FAISS index.")
