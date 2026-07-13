#!/usr/bin/env bash
# deploy.sh — 重力知识库一键启动脚本 (Linux / macOS)
# 用法: chmod +x deploy.sh && ./deploy.sh

set -e

PORT=8765

info()  { echo -e "\033[36m==>\033[0m $1"; }
ok()    { echo -e "  \033[32m[OK]\033[0m $1"; }
fail()  { echo -e "  \033[31m[FAIL]\033[0m $1"; }

echo ""
echo -e "\033[35m  重力知识库 - Gravity Knowledge Base\033[0m"
echo -e "\033[35m  ====================================\033[0m"
echo ""

# 1. 检查 Python
info "检查 Python 环境..."
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    fail "Python 未安装。"
    exit 1
fi
ok "Python: $($PY --version)"

# 2. 检查 Ollama
info "检查 Ollama 服务..."
if curl -sf http://localhost:11434 > /dev/null 2>&1; then
    ok "Ollama 服务运行中"
else
    fail "Ollama 未运行。请先启动 Ollama。"
    exit 1
fi

# 3. 检查嵌入模型
info "检查嵌入模型 (nomic-embed-text)..."
if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
    ok "嵌入模型已就绪"
else
    echo -e "  \033[33m[WARN]\033[0m nomic-embed-text 未找到，正在下载..."
    ollama pull nomic-embed-text
    ok "嵌入模型下载完成"
fi

# 4. 安装依赖
info "安装 Python 依赖..."
cd "$(dirname "$0")"
pip install -q -r requirements.txt
ok "依赖安装完成"

# 5. 清理旧进程
info "清理端口 $PORT..."
old_pid=$(lsof -ti :$PORT 2>/dev/null || true)
if [ -n "$old_pid" ]; then
    kill -9 "$old_pid" 2>/dev/null || true
    sleep 1
    ok "已释放端口 $PORT (旧 PID: $old_pid)"
fi

# 6. 启动服务
info "启动知识库服务..."
LOG_FILE="$(pwd)/kb_server.log"
$PY -X utf8 -m uvicorn api.main:app --host 0.0.0.0 --port "$PORT" --log-level info > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

sleep 3

# 验证启动
if curl -sf "http://127.0.0.1:${PORT}/status" > /dev/null 2>&1; then
    ok "服务已就绪 (PID=$SERVER_PID)"
    echo ""
    echo -e "\033[36m  打开浏览器访问:\033[0m"
    echo "  http://localhost:${PORT}"
    echo ""
    echo -e "\033[90m  日志文件: $LOG_FILE\033[0m"
    echo ""
    
    # 尝试打开浏览器
    if command -v xdg-open &>/dev/null; then
        xdg-open "http://localhost:${PORT}" 2>/dev/null
    elif command -v open &>/dev/null; then
        open "http://localhost:${PORT}" 2>/dev/null
    fi
else
    fail "服务启动失败，请查看日志: $LOG_FILE"
    exit 1
fi
