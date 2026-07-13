# deploy.ps1 — 重力知识库一键启动脚本 (Windows PowerShell)
# 用法: powershell -ExecutionPolicy Bypass -File deploy.ps1

$ErrorActionPreference = "Stop"
$Port = 8765

function Write-Step {
    param([string]$Msg)
    Write-Host "==> $Msg" -ForegroundColor Cyan
}

function Write-OK {
    param([string]$Msg)
    Write-Host "  [OK] $Msg" -ForegroundColor Green
}

function Write-Fail {
    param([string]$Msg)
    Write-Host "  [FAIL] $Msg" -ForegroundColor Red
}

Write-Host ""
Write-Host "  重力知识库 - Gravity Knowledge Base" -ForegroundColor Magenta
Write-Host "  ====================================" -ForegroundColor Magenta
Write-Host ""

# --- 1. 检查 Python ---
Write-Step "检查 Python 环境..."
try {
    $pyVersion = python --version 2>&1
    Write-OK "Python: $pyVersion"
} catch {
    Write-Fail "Python 未安装或不在 PATH 中"
    exit 1
}

# --- 2. 检查 Ollama ---
Write-Step "检查 Ollama 服务..."
try {
    $ollamaStatus = curl.exe -s -o NUL -w "%{http_code}" http://localhost:11434
    if ($ollamaStatus -eq "200") {
        Write-OK "Ollama 服务运行中"
    } else {
        throw "Ollama 响应异常: $ollamaStatus"
    }
} catch {
    Write-Fail "Ollama 未运行。请先启动 Ollama 后再运行此脚本。"
    exit 1
}

# --- 3. 检查嵌入模型 ---
Write-Step "检查嵌入模型 (nomic-embed-text)..."
try {
    $models = curl.exe -s http://localhost:11434/api/tags | python -c "import sys,json; d=json.load(sys.stdin); print('\n'.join(m['name'] for m in d.get('models',[])))" 2>&1
    if ($models -match "nomic-embed-text") {
        Write-OK "嵌入模型已就绪"
    } else {
        Write-Host "  [WARN] nomic-embed-text 未找到，正在下载..." -ForegroundColor Yellow
        ollama pull nomic-embed-text
        Write-OK "嵌入模型下载完成"
    }
} catch {
    Write-Fail "无法查询 Ollama 模型列表: $_"
    exit 1
}

# --- 4. 安装依赖 ---
Write-Step "安装 Python 依赖..."
$requirements = Join-Path $PSScriptRoot "requirements.txt"
if (Test-Path $requirements) {
    pip install -q -r $requirements 2>&1 | Out-Null
    Write-OK "依赖安装完成"
} else {
    Write-Fail "requirements.txt 未找到"
    exit 1
}

# --- 5. 清理旧进程 ---
Write-Step "清理端口 $Port ..."
$oldProc = netstat -ano | Select-String ":${Port}\s" | Select-String "LISTENING"
if ($oldProc) {
    $oldPid = ($oldProc -split "\s+")[-1]
    try {
        Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
        Write-OK "已释放端口 $Port (旧 PID: $oldPid)"
    } catch {
        Write-Host "  [WARN] 无法终止旧进程: $_" -ForegroundColor Yellow
    }
}

# --- 6. 启动服务 ---
Write-Step "启动知识库服务..."
$serverDir = $PSScriptRoot
$logFile = Join-Path $serverDir "kb_server.log"

try {
    $process = Start-Process -NoNewWindow -PassThru -FilePath "python" -ArgumentList "-X utf8", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", $Port, "--log-level", "info" `
        -WorkingDirectory $serverDir -RedirectStandardOutput $logFile -RedirectStandardError $logFile
    
    Start-Sleep -Seconds 3
    
    # 验证启动
    try {
        $check = curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:${Port}/status
        if ($check -eq "200") {
            Write-OK "服务已就绪 (PID=$($process.Id))"
            Write-Host ""
            Write-Host "  打开浏览器访问:" -ForegroundColor Cyan
            Write-Host "  http://localhost:${Port}" -ForegroundColor White
            Write-Host ""
            Write-Host "  日志文件: $logFile" -ForegroundColor Gray
            Write-Host ""
            
            # 打开浏览器
            Start-Process "http://localhost:${Port}"
        } else {
            Write-Fail "服务启动异常 (HTTP $check)"
            exit 1
        }
    } catch {
        Write-Fail "服务启动失败，请查看日志: $logFile"
        exit 1
    }
} catch {
    Write-Fail "启动失败: $_"
    exit 1
}

# 保持窗口打开
Write-Host "按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
