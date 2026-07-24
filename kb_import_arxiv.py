"""
usage: python kb_import_arxiv.py [category1 category2 ...]
从 arXiv API 批量抓取论文摘要 → 灌入 local-kb API.

默认抓取 cs.AI, cs.LG, math.IT 最新论文。
每个摘要存为单文件 .md，再上传。

前置条件:
  - local-kb API 运行在 http://127.0.0.1:8766
  - pip install httpx

示例:
  python kb_import_arxiv.py cs.CL cs.CV          # NLP + CV
  python kb_import_arxiv.py --all                  # 所有 CS 子领域
  python kb_import_arxiv.py --max 500              # 每类最多 500 篇
"""

import httpx, os, sys, time, json, hashlib
from datetime import datetime, timezone
from pathlib import Path

API = "http://127.0.0.1:8766"
CACHE = Path(__file__).parent / ".arxiv_cache"
os.makedirs(CACHE, exist_ok=True)

# 默认类别
DEFAULT_CATS = ["cs.AI", "cs.LG", "ms.IT", "math.NA"]

# arXiv API URL
ARXIV_URL = "http://export.arxiv.org/api/query"


def fetch_papers(category: str, max_results: int = 100) -> list[dict]:
    """从 arXiv API 抓取最新论文"""
    query = f"cat:{category}"
    params = f"search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    url = f"{ARXIV_URL}?{params}"

    print(f"  Fetching {category} (max={max_results})...", end=" ", flush=True)
    try:
        r = httpx.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"HTTP error: {e}")
        return []

    import xml.etree.ElementTree as ET
    ns = {"a": "http://www.w3.org/2005/Atom",
          "arxiv": "http://arxiv.org/schemas/atom"}

    papers = []
    root = ET.fromstring(r.text)
    for entry in root.findall("a:entry", ns):
        paper_id = entry.find("a:id", ns).text.strip()
        title = entry.find("a:title", ns).text.strip().replace("\n", " ").replace("  ", " ")
        summary = entry.find("a:summary", ns).text.strip().replace("\n", " ").replace("  ", " ")
        published = entry.find("a:published", ns).text.strip()[:10]

        authors = []
        for author in entry.findall("a:author", ns):
            name = author.find("a:name", ns)
            if name is not None:
                authors.append(name.text.strip())

        papers.append({
            "id": paper_id,
            "title": title,
            "summary": summary,
            "authors": ", ".join(authors[:5]) + ("..." if len(authors) > 5 else ""),
            "date": published,
            "categories": [cat.text for cat in entry.findall("a:category", ns)],
        })

    print(f"{len(papers)} papers")
    return papers


def paper_to_md(paper: dict) -> tuple[str, str]:
    """论文 → .md 文件 (filename, content)"""
    # 用 paper ID 的 hash 做文件名
    fid = hashlib.md5(paper["id"].encode()).hexdigest()[:12]
    filename = f"arxiv_{paper['date']}_{fid}.md"

    categories = ", ".join(paper["categories"][:5])
    content = f"""# {paper['title']}

- **arXiv ID:** `{paper['id'].split('/')[-1].split('v')[0]}`
- **Published:** {paper['date']}
- **Categories:** {categories}
- **Authors:** {paper['authors']}

## Abstract

{paper['summary']}

---
*Auto-imported from arXiv on {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
    return filename, content


def upload_md(filename: str, content: str) -> int:
    """上传 .md 内容到 local-kb"""
    try:
        r = httpx.post(
            f"{API}/upload",
            files={"file": (filename, content.encode("utf-8"), "text/markdown")},
            data={"source_type": "学术论文", "auto_rebuild": "false"},
            timeout=180,
        )
        if r.status_code == 200:
            return r.json().get("new_spheres", 0)
        else:
            return f"ERR{r.status_code}"
    except Exception as e:
        return str(type(e).__name__)


def main():
    categories = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CATS
    max_results = 100

    # Parse --max
    if "--max" in categories:
        idx = categories.index("--max")
        max_results = int(categories.pop(idx + 1))
        categories.pop(idx)
    
    if "--all" in categories:
        categories = [
            "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR",
            "cs.CV", "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS",
            "cs.ET", "cs.FL", "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR",
            "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", "cs.NA",
            "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO",
            "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY",
            "math.IT", "math.NA", "math.OC", "math.ST",
            "stat.ML", "stat.TH",
        ]

    # Check API
    try:
        r = httpx.get(f"{API}/status", timeout=5)
        d = r.json()
        print(f"API OK: {d['active_spheres']} spheres, {d['faiss_vectors']} vectors\n")
    except Exception as e:
        print(f"API not reachable: {e}")
        sys.exit(1)

    total_papers = 0
    total_new = 0

    for cat in categories:
        cat = cat.strip()
        if not cat:
            continue
        papers = fetch_papers(cat, max_results)
        if not papers:
            continue

        for i, paper in enumerate(papers):
            filename, content = paper_to_md(paper)
            result = upload_md(filename, content)
            if isinstance(result, int):
                total_new += result
            status = f"+{result}" if isinstance(result, int) else str(result)
            print(f"    [{i+1}/{len(papers)}] {paper['title'][:55]:55s} {status}")
            total_papers += 1

            if (i + 1) % 20 == 0:
                print(f"    → {total_papers} papers, +{total_new} spheres so far")
                time.sleep(1)

        time.sleep(2)  # 类别间间隔

    print(f"\nDone: {total_papers} papers, +{total_new} new spheres")
    print(f"Total API status:")
    r = httpx.get(f"{API}/status", timeout=10)
    print(r.json())


if __name__ == "__main__":
    main()
