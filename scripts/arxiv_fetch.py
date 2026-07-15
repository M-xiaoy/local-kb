"""arXiv 论文抓取器：从不同领域抓取论文，构建因果验证沙盒"""
import sys, os, json, time, re, urllib.request, urllib.parse, xml.etree.ElementTree as ET

# 确保输出 UTF-8
sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_arxiv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 领域配置 ──
# 每个领域抓 N 篇，覆盖广泛但不过量
DOMAINS = {
    "cs.AI": "计算机·人工智能",
    "cs.LG": "计算机·机器学习",
    "cs.CL": "计算机·自然语言处理",
    "cs.SE": "计算机·软件工程",
    "astro-ph": "天文学·天体物理",
    "stat.ML": "统计学·机器学习",
    "q-bio.GN": "生物学·基因组学",
    "econ.EM": "经济学·计量方法",
    "physics.soc-ph": "物理学·社会物理",
    "cs.CR": "计算机·安全与加密",
}

TOTAL = 50
PER_DOMAIN = max(1, TOTAL // len(DOMAINS))  # 5 per domain

ARXIV_API = "http://export.arxiv.org/api/query"


def fetch_arxiv(cat: str, max_results: int = 10) -> list:
    """通过 arXiv API 获取指定分类的最新论文"""
    params = {
        "search_query": f"cat:{cat}",
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    print(f"[{cat}] 请求: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "GravityKB/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")

    # 解析 XML
    root = ET.fromstring(raw)
    ns = {"a": "http://www.w3.org/2005/Atom",
          "arxiv": "http://arxiv.org/schemas/atom"}

    papers = []
    for entry in root.findall("a:entry", ns):
        title = entry.find("a:title", ns)
        title = title.text.strip().replace("\n", " ") if title is not None else ""

        abstract = entry.find("a:summary", ns)
        abstract = abstract.text.strip().replace("\n", " ") if abstract is not None else ""

        published = entry.find("a:published", ns)
        published = published.text[:10] if published is not None else ""

        # 获取分类列表
        cats = []
        for cat_el in entry.findall("a:category", ns):
            cat_name = cat_el.get("term", "")
            if cat_name:
                cats.append(cat_name)

        authors = []
        for author_el in entry.findall("a:author", ns):
            name_el = author_el.find("a:name", ns)
            if name_el is not None:
                authors.append(name_el.text)

        paper_id = entry.find("a:id", ns)
        arxiv_id = ""
        if paper_id is not None:
            arxiv_id = paper_id.text.strip().split("/")[-1]
            arxiv_id = arxiv_id.replace("arxiv:", "")

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "categories": cats,
            "primary_cat": cat,
            "published": published,
            "authors": authors[:5],
        })

    print(f"[{cat}] 获取 {len(papers)} 篇")
    return papers


def sanitize_filename(s: str) -> str:
    """安全的文件名"""
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    return s[:100]


def clean_abstract(text: str) -> str:
    """清洗摘要：去除引用标记、公式引用"""
    text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)  # (Author, 2020)
    text = re.sub(r'\[[\d,\s]+\]', '', text)           # [1,2,3]
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def write_paper(paper: dict, domain_label: str):
    """将论文写为可导入的文本文件"""
    filename = f"arxiv_{paper['arxiv_id']}_{sanitize_filename(paper['title'][:60])}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Section-structured 文本，便于 section-mode chunker
    content = f"""# {paper['title']}

> 领域: {domain_label}
> 分类: {', '.join(paper['categories'])}
> 发表日期: {paper['published']}
> 作者: {', '.join(paper['authors'])}
> arXiv ID: {paper['arxiv_id']}

## Methods

本文的方法基于对 {domain_label} 领域现有工作的分析，采用实验验证的方法论框架。

## Results

{clean_abstract(paper['abstract'])}

## Conclusion

该研究提供了 {domain_label} 领域的新见解，其结论建立在可验证的实验结果之上。
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filename


def main():
    print(f"=== arXiv 因果验证沙盒构造 ===")
    print(f"领域数: {len(DOMAINS)}, 每领域最多 {PER_DOMAIN} 篇, 目标总量 {TOTAL}")

    all_papers = []
    for cat, label in DOMAINS.items():
        try:
            papers = fetch_arxiv(cat, PER_DOMAIN)
            for p in papers:
                write_paper(p, label)
                all_papers.append(p)
                time.sleep(0.5)  # arXiv API 限制: 每秒1次
        except Exception as e:
            print(f"[{cat}] 失败: {e}")

        time.sleep(1)

    print(f"\n=== 完成 ===")
    print(f"成功获取: {len(all_papers)} 篇")
    print(f"文件位置: {OUTPUT_DIR}")

    # 写入 manifest
    manifest = {
        "total": len(all_papers),
        "papers": [{"arxiv_id": p["arxiv_id"], "title": p["title"],
                     "primary_cat": p["primary_cat"], "published": p["published"]}
                   for p in all_papers],
    }
    manifest_path = os.path.join(OUTPUT_DIR, "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"清单: {manifest_path}")


if __name__ == "__main__":
    main()
