# 已迁移至 analysis/deprecated_pipeline/role_expander.py (2026-07-24)
# 原因：RoleExpander 依赖 RoleTable 1.6GB 加载，不参与实时检索。
raise RuntimeError("RoleExpander 已移出检索链路。请从 analysis/deprecated_pipeline/ 导入。")
