"""
kb_ctl.py — 知识库服务控制
==========================
用法：
  python kb_ctl.py start    # 启动（如果已有健康服务则跳过）
  python kb_ctl.py stop     # 停止
  python kb_ctl.py status   # 查看状态
  python kb_ctl.py restart  # 重启（先停再起）
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HOST = "0.0.0.0"
PORT = 8765
API_ROOT = Path(__file__).parent  # local-kb 目录
PID_FILE = API_ROOT / ".kb_pid"
LOG_FILE = API_ROOT / "kb_server.log"


# ──────────────────────────────────────────────
# 端口 / 进程工具
# ──────────────────────────────────────────────

def is_port_in_use(port: int) -> bool:
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def is_server_healthy() -> bool:
    """检查服务是否正常响应

    用 urllib.request 而非 httpx/requests，原因：
    httpx 默认 trust_env=True，在 Windows 上会读取系统代理设置
    （Internet Options 中的代理），导致本地 health check 请求被
    发到代理服务器（如 127.0.0.1:80），而非目标服务。
    urllib.request 不继承系统代理，本地直连，不会误判。
    """
    if not is_port_in_use(PORT):
        return False
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/status", timeout=3
        )
        return resp.status == 200
    except Exception:
        return False


def find_server_pid() -> int | None:
    """通过 PID 文件查找进程"""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)  # 信号 0 = 仅检查存在性
            return pid
        except (OSError, ProcessLookupError):
            PID_FILE.unlink(missing_ok=True)
            return None
    return None


def find_uvicorn_pids() -> list[int]:
    """扫描所有 uvicorn 进程"""
    import psutil
    pids = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "uvicorn" in cmd.lower() and str(PORT) in cmd:
                pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def kill_pid(pid: int) -> bool:
    """安全杀进程"""
    try:
        os.kill(pid, signal.SIGTERM)
        # 给 3 秒优雅退出
        for _ in range(6):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except (OSError, ProcessLookupError):
                return True
        # 超时了强制杀
        os.kill(pid, signal.SIGKILL)
        return True
    except (OSError, ProcessLookupError):
        return True
    except Exception as e:
        print(f"  [!] 杀进程失败 PID={pid}: {e}")
        return False


# ──────────────────────────────────────────────
# 命令实现
# ──────────────────────────────────────────────

def cmd_status():
    print(f"  端口 {PORT}: {'占用中' if is_port_in_use(PORT) else '空闲'}")
    pid = find_server_pid()
    print(f"  PID 文件: {'有 (PID=' + str(pid) + ')' if pid else '无'}")

    if is_server_healthy():
        import urllib.request, json
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/status", timeout=3
        )
        s = json.loads(resp.read())
        print(f"  ─────────────────────")
        print(f"  服务状态: ✅ 正常运行")
        print(f"  活跃球体: {s['active_spheres']}")
        print(f"  FAISS 向量: {s['faiss_vectors']}")
        print(f"  场域: {', '.join(s['fields']) or '无'}")
    else:
        print(f"  服务状态: {'❌ 端口占用但无响应' if is_port_in_use(PORT) else '⏹️ 未启动'}")


def cmd_stop():
    print(f"  停止知识库服务 (端口 {PORT})...")

    # 1. 通过 PID 文件杀
    pid = find_server_pid()
    if pid:
        print(f"  找到 PID 文件: {pid}")
        kill_pid(pid)
        PID_FILE.unlink(missing_ok=True)

    # 2. 扫描残留 uvicorn 进程
    leftovers = find_uvicorn_pids()
    for p in leftovers:
        print(f"  清理残留进程: PID={p}")
        kill_pid(p)

    print(f"  已停止")


def cmd_start():
    if is_server_healthy():
        print(f"  服务已在运行 (端口 {PORT})，跳过启动")
        cmd_status()
        return

    # 如果端口被占但服务不健康 → 清理
    if is_port_in_use(PORT):
        print(f"  端口 {PORT} 被占用但服务无响应，正在清理...")
        cmd_stop()
        time.sleep(1)

    print(f"  启动知识库服务 (port={PORT})...")

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n--- START {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app",
             "--host", HOST, "--port", str(PORT)],
            cwd=str(API_ROOT),
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

        # 写 PID 文件
        PID_FILE.write_text(str(proc.pid))

    # 等待服务就绪（最多 30 秒）
    for i in range(30):
        if is_server_healthy():
            print(f"  ✅ 服务已就绪 (PID={proc.pid})")
            # 打印启动日志最后几行
            try:
                tail = open(LOG_FILE, encoding="utf-8", errors="replace").read().strip().split("\n")[-5:]
                for line in tail:
                    if any(kw in line.lower() for kw in ["info", "error", "warning", "started", "loaded"]):
                        print(f"    {line.strip()}")
            except Exception:
                pass  # 日志读取失败不影响服务运行
            return
        time.sleep(1)

    print(f"  ⚠️ 服务启动超时，请检查日志: {LOG_FILE}")


def cmd_restart():
    cmd_stop()
    time.sleep(1)
    cmd_start()


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"

    if action == "start":
        cmd_start()
    elif action == "stop":
        cmd_stop()
    elif action == "restart":
        cmd_restart()
    elif action == "status":
        cmd_status()
    else:
        print(f"未知命令: {action}")
        print(f"用法: python kb_ctl.py [start|stop|restart|status]")
        sys.exit(1)
