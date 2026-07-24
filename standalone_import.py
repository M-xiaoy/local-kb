"""
standalone_import.py — 不依赖 API，直连 KB 核心模块批量灌库

用法:
  python standalone_import.py                     # GitHub + arXiv
  python standalone_import.py --github-only
  python standalone_import.py --arxiv-cats cs.CL
  python standalone_import.py --rebuild-only       # 只重建索引不导入
"""

import hashlib, json, logging, os, subprocess, sys, time, xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

KB_DIR = Path(__file__).parent.resolve()
os.chdir(str(KB_DIR))
sys.path.insert(0, str(KB_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("standalone_import")

# ─── Config ───────────────────────────────────
GITHUB_REPOS = [
    "https://github.com/datawhalechina/thorough-pytorch.git",
    "https://github.com/datawhalechina/happy-llm.git",
    "https://github.com/datawhalechina/leedl-tutorial.git",
    "https://github.com/PKUFlyingPig/cs-self-learning.git",
    "https://github.com/RiazML/math-for-llms.git",
]
ARXIV_CATS = ["cs.AI", "cs.LG", "math.NA", "cs.CL", "cs.CV", "cs.IR", "cs.NE"]
ARXIV_MAX = 100
EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".github",
                "images", "img", "assets", "fonts", ".venv", "dist", "build"}

# ─── Sleep Prevention ─────────────────────────
def keep_awake(on=True):
    try:
        val = "0" if on else "30"
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", val], capture_output=True, timeout=5)
        subprocess.run(["powercfg", "/change", "hibernate-timeout-ac", val], capture_output=True, timeout=5)
        logger.info(f"Sleep {'disabled' if on else 'restored to 30min'}")
    except Exception as e:
        logger.warning(f"powercfg: {e}")

# ─── KB Bootstrap ─────────────────────────────
def bootstrap_kb():
    """直接构造 KB，跳过 API 和 startup event"""
    from config import paths as cfg_paths
    from storage.sphere_store import SphereStore
    from storage.faiss_store import FaissStore
    from storage.registry import Registry
    from pipeline.embedder import Embedder
    from core.repo.adapter import AdapterRepository
    from core.kb import KnowledgeBase

    logger.info("Loading existing data...")
    t0 = time.time()

    ss = SphereStore(str(cfg_paths.spheres_data))
    fs = FaissStore(str(cfg_paths.faiss_index))
    reg = Registry(str(cfg_paths.registry_map))
    embedder = Embedder()

    ss.load()
    reg.load()
    fs.load()
    logger.info(f"  {ss.count} spheres, {reg.count} mappings, {fs.count} vectors ({time.time()-t0:.1f}s)")

    repo = AdapterRepository(ss, fs, reg)
    kb = KnowledgeBase(repo=repo, embedder=embedder)
    logger.info(f"KB ready")
    return kb, ss, fs, reg

# ─── GitHub ───────────────────────────────────
def clone(url: str):
    name = url.split("/")[-1].replace(".git", "")
    target = KB_DIR / ".repo_cache" / name
    if target.exists():
        logger.info(f"  [SKIP] {name}")
        return name, target
    logger.info(f"  [CLONE] {name} ...")
    try:
        r = subprocess.run(["git", "clone", "--depth", "1", url, str(target)], capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            return name, target
        logger.warning(f"  [FAIL] {name}: {r.stderr[:100]}")
        return None, None
    except subprocess.TimeoutExpired:
        logger.warning(f"  [FAIL] {name}: timeout")
        return None, None

def scan_md(dir_path: Path):
    files = []
    for root, dirs, _ in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in Path(root).glob("*.md"):
            if f.stat().st_size >= 500:
                files.append(f)
    return sorted(files)

def import_github(kb) -> int:
    logger.info("\n=== GitHub ===")
    total_new = 0; total_files = 0
    os.makedirs(KB_DIR / ".repo_cache", exist_ok=True)
    for url in GITHUB_REPOS:
        name, target = clone(url)
        if target is None: continue
        mds = scan_md(target)
        if not mds: continue
        logger.info(f"  {name}: {len(mds)} files")
        for md in mds:
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
                n = kb.add_document(text=text, source_file=str(md), source_type="技术笔记")
                total_new += n; total_files += 1
                if total_files % 20 == 0:
                    logger.info(f"    ... {total_files} files done, +{total_new} spheres")
            except Exception as e:
                logger.warning(f"    SKIP {md.name}: {e}")
        logger.info(f"  {name}: done")
    logger.info(f"GitHub: +{total_new} from {total_files} files")
    return total_new

# ─── arXiv ────────────────────────────────────
def fetch_arxiv(cat: str, max_r: int = 100):
    import httpx
    url = f"http://export.arxiv.org/api/query?search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending&max_results={max_r}"
    try:
        r = httpx.get(url, timeout=60); r.raise_for_status()
    except Exception as e:
        logger.warning(f"  Fetch {cat} error: {e}"); return []
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    papers = []
    root = ET.fromstring(r.text)
    for entry in root.findall("a:entry", ns):
        papers.append({
            "id": entry.find("a:id", ns).text.strip(),
            "title": entry.find("a:title", ns).text.strip().replace("\n"," ").replace("  "," "),
            "summary": entry.find("a:summary", ns).text.strip().replace("\n"," ").replace("  "," "),
            "date": entry.find("a:published", ns).text.strip()[:10],
        })
    return papers

def import_arxiv(kb) -> int:
    logger.info("\n=== arXiv ===")
    total_new = 0; total_papers = 0
    for cat in ARXIV_CATS:
        papers = fetch_arxiv(cat, ARXIV_MAX)
        if not papers: continue
        logger.info(f"  {cat}: {len(papers)} papers")
        for paper in papers:
            try:
                content = f"# {paper['title']}\n\n{paper['summary']}"
                doc_id = f"arxiv-{hashlib.md5(paper['id'].encode()).hexdigest()[:12]}"
                n = kb.add_document(text=content, source_file=doc_id, source_type="学术论文")
                total_new += n; total_papers += 1
            except: pass
        time.sleep(2)
    logger.info(f"arXiv: +{total_new} from {total_papers} papers")
    return total_new

# ─── Rebuild ──────────────────────────────────
def do_rebuild(ss, fs, reg):
    logger.info("\n=== Rebuild ===")
    t0 = time.time()
    from config import paths as cfg_paths
    from storage.calibrator import SphereCalibrator
    from storage.sphere_store import SphereStore, mass_to_norm
    from pipeline.embedder import poincare_project
    import numpy as np

    calibrator = SphereCalibrator()
    calibrator.attach(ss, fs._vectors)
    cal_stats = calibrator.calibrate_all()
    logger.info(f"Calibrated: +{cal_stats.get('new_connections',0)} connections")

    # Re-poincare
    active = ss.get_active()
    norms = np.array([mass_to_norm(s.mass) for s in active], dtype=np.float32)
    vectors_list = []
    ids_list = []
    for s in active:
        fid = reg.faiss_id(s.id)
        if fid is not None and fid in fs._vectors:
            vectors_list.append(fs._vectors[fid])
            ids_list.append(fid)
    if vectors_list:
        vectors = np.stack(vectors_list, axis=0)
        poincare_vecs = poincare_project(vectors, norms)
        for i, fid in enumerate(ids_list):
            fs._vectors[fid] = poincare_vecs[i]

    # Rebuild FAISS
    fs.reset()
    fs.add(np.stack(list(fs._vectors.values()), axis=0), np.array(list(fs._vectors.keys()), dtype=np.int64))
    fs.save()
    ss.save()
    reg.save()
    logger.info(f"  Rebuild done ({time.time()-t0:.1f}s)")

# ─── Main ─────────────────────────────────────
def main():
    args = set(sys.argv[1:])

    if "--rebuild-only" in args:
        kb, ss, fs, reg = bootstrap_kb()
        do_rebuild(ss, fs, reg)
        return

    do_github = "--arxiv-only" not in args
    do_arxiv = "--github-only" not in args

    if "--arxiv-cats" in args:
        idx = list(args).index("--arxiv-cats")
        global ARXIV_CATS
        ARXIV_CATS = list(args)[idx + 1:]

    logger.info("=" * 50)
    logger.info(f"  standalone_import — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 50)

    keep_awake(True)
    t0 = time.time()
    try:
        kb, ss, fs, reg = bootstrap_kb()
        if do_github: import_github(kb)
        if do_arxiv: import_arxiv(kb)
        logger.info("Saving...")
        ss.save()
        fs.save()
        reg.save()
        do_rebuild(ss, fs, reg)
        logger.info(f"\nDONE! {ss.count} spheres, {time.time()-t0:.0f}s ({((time.time()-t0)/60):.1f}min)")
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
    finally:
        keep_awake(False)

if __name__ == "__main__":
    main()
