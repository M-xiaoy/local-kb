import sys, json
sys.stdout.reconfigure(encoding="utf-8")
with open("data/spheres/spheres.json","r",encoding="utf-8") as f:
    d=json.load(f)
spheres = list(d.get("spheres",{}).values())

# 连接最多的 5 个球体
by_degree = sorted(spheres, key=lambda s: len(s.get("connections",{})), reverse=True)[:5]
print("=== deg/mass/connections ===")
for s in by_degree:
    deg = len(s.get("connections",{}))
    m = s.get("effective_mass",1.0)
    src = s.get("source_type","?")
    txt = s.get("text","")[:80]
    print(f"deg={deg:3d}  mass={m:.3f}  [{src}]  {txt}")

# 总连接数
total_edges = sum(len(s.get("connections",{})) for s in spheres)
print(f"\n总球体: {len(spheres)}")
print(f"总连接数(双向): {total_edges}")
print(f"平均 degree: {total_edges/len(spheres):.1f}")

# 查看几条具体连接
print("\n=== 样本连接 ===")
for s in spheres:
    conns = list(s.get("connections",{}).items())
    if len(conns) >= 3:
        print(f"\n球体({s.get('text','')[:50]}...) 的连接:")
        for cid, cdata in conns[:3]:
            print(f"  -> {cdata.get('target','?'):.50s}  w={cdata.get('weight',0):.3f}  type={cdata.get('type','dendrite')}")
        print(f"  ... 共 {len(conns)} 条连接")
        break
