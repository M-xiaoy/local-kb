"""快速导入所有已有文件到 local-kb"""
import os, sys, time, httpx

SERVER = 'http://127.0.0.1:8766'
DATA_DIR = r'C:\Users\lc202\.openclaw\workspace\local-kb\data\uploads'

# 先检查服务
try:
    r = httpx.get(f'{SERVER}/status', timeout=5)
    print(f'服务状态: ✅ 在线，当前 {r.json()["active_spheres"]} 球体')
except Exception as e:
    print(f'服务状态: ❌ {e}')
    sys.exit(1)

# 找所有文件
files = []
for f in sorted(os.listdir(DATA_DIR)):
    if f.endswith(('.md', '.txt', '.json', '.py')):
        files.append(os.path.join(DATA_DIR, f))

print(f'找到 {len(files)} 个文件，开始导入...')

success = 0
failed = 0
for i, fp in enumerate(files):
    fname = os.path.basename(fp)
    fsize = os.path.getsize(fp)
    
    # 推断类型
    ext = os.path.splitext(fname)[1].lower()
    type_map = {'.md': '日记', '.txt': '笔记', '.json': '配置', '.py': '代码'}
    src_type = type_map.get(ext, '其他')
    
    print(f'[{i+1}/{len(files)}] {fname} ({fsize//1024}KB, {src_type})', end='')
    
    try:
        with open(fp, 'rb') as f:
            r = httpx.post(
                f'{SERVER}/upload',
                files={'file': (fname, f, f'text/{ext[1:] if ext[1:] else "plain"}')},
                data={'source_type': src_type},
                timeout=60
            )
        if r.status_code == 200:
            data = r.json()
            chunks = data.get('chunks', 0)
            print(f' ✅ {chunks} chunks')
            success += 1
        else:
            print(f' ❌ HTTP {r.status_code}')
            failed += 1
    except Exception as e:
        print(f' ❌ {str(e)[:60]}')
        failed += 1
    
    # 每5个文件休息一下，给embedder喘口气
    if (i + 1) % 5 == 0:
        time.sleep(1)

print(f'\n=== 导入完成 ===')
print(f'成功: {success} / {len(files)}, 失败: {failed}')

# 查一下最终状态
r = httpx.get(f'{SERVER}/status', timeout=5)
s = r.json()
print(f'最终状态: 球体 {s["active_spheres"]} / FAISS向量 {s["faiss_vectors"]}')
