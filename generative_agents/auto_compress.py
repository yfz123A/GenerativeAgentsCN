#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""定时把挂机产出的存档压成录像，让直播台跟上进度。

用法（在 generative_agents 目录下）：

    setsid nohup /root/venvs/ga_cn/bin/python auto_compress.py --name live-town \
        > /tmp/auto_compress.log 2>&1 < /dev/null &

行为：
  - 每 --interval 秒检查一次，**只在「有比上次压缩更新的存档」时才真正跑 compress.py**
    （用文件 mtime 判断，不读大文件）；
  - compress 失败不影响挂机，只记日志并在下一轮重试；
  - 单实例锁（.autocompress.pid），避免重复跑；
  - 直播台是「固定时间轴 + 一次性注入」：压缩完观众需要刷新页面才会拿到新进度。
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

STOPPING = False


def log(msg):
    print("[auto-compress {}] {}".format(datetime.now().strftime("%m-%d %H:%M:%S"), msg), flush=True)


def newest_mtime(folder, prefix="simulate-", suffix=".json"):
    """目录里最新存档的修改时间（没有则返回 0）。"""
    newest = 0.0
    try:
        for name in os.listdir(folder):
            if name.startswith(prefix) and name.endswith(suffix):
                newest = max(newest, os.path.getmtime(os.path.join(folder, name)))
    except OSError:
        return 0.0
    return newest


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def on_signal(signum, frame):
    global STOPPING
    STOPPING = True
    log("收到信号 {}，退出。".format(signum))


def main():
    ap = argparse.ArgumentParser(description="定时压缩挂机存档（供直播台播放）")
    ap.add_argument("--name", type=str, required=True, help="模拟名称")
    ap.add_argument("--interval", type=float, default=1800.0, help="检查间隔秒数，默认 1800（30 分钟）")
    ap.add_argument("--min-steps", type=int, default=1, help="至少有多少步新存档才压缩（默认 1）")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    ckpt_dir = os.path.join("results", "checkpoints", args.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    movement = os.path.join("results", "compressed", args.name, "movement.json")

    # 单实例锁
    lock_path = os.path.join(ckpt_dir, ".autocompress.pid")
    if os.path.exists(lock_path):
        try:
            old = int(open(lock_path, encoding="utf-8").read().strip())
        except (ValueError, OSError):
            old = 0
        if old and old != os.getpid() and pid_alive(old):
            log("已有自动压缩在跑（pid {}），退出。".format(old))
            return 1
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    log("自动压缩启动：name={} 间隔={:.0f}s".format(args.name, args.interval))
    rounds = 0
    while not STOPPING:
        rounds += 1
        ckpt_mtime = newest_mtime(ckpt_dir)
        done_mtime = os.path.getmtime(movement) if os.path.exists(movement) else 0.0

        if ckpt_mtime <= done_mtime:
            log("第 {} 轮：没有新存档（存档 {} / 录像 {}），跳过".format(
                rounds,
                datetime.fromtimestamp(ckpt_mtime).strftime("%H:%M:%S") if ckpt_mtime else "无",
                datetime.fromtimestamp(done_mtime).strftime("%H:%M:%S") if done_mtime else "无",
            ))
        else:
            log("第 {} 轮：发现新存档，开始压缩…".format(rounds))
            started = time.time()
            try:
                p = subprocess.run([sys.executable, "compress.py", "--name", args.name],
                                   cwd=here, capture_output=True, text=True, timeout=1800)
                if p.returncode == 0:
                    size = os.path.getsize(movement) / (1024 ** 2) if os.path.exists(movement) else 0
                    log("压缩完成，用时 {:.1f}s，录像 {:.1f}MB（观众刷新页面即可看到新进度）".format(
                        time.time() - started, size))
                else:
                    log("compress 失败（退出码 {}）：{}".format(
                        p.returncode, (p.stderr or p.stdout or "").strip()[-300:]))
            except subprocess.TimeoutExpired:
                log("compress 超时（30 分钟），跳过这一轮")
            except Exception as e:      # 任何异常都不该影响挂机
                log("compress 异常：{}".format(e))

        # 分片睡眠，便于快速响应停止信号
        slept = 0.0
        while slept < args.interval and not STOPPING:
            time.sleep(min(5.0, args.interval - slept))
            slept += 5.0

    try:
        os.remove(lock_path)
    except OSError:
        pass
    log("自动压缩退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
