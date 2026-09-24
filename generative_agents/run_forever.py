#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""挂机监督器：把 `start.py --forever --resume` 反复拉起，直到正常结束或磁盘不足。

用法（在 generative_agents 目录下）：

    setsid nohup /root/venvs/ga_cn/bin/python run_forever.py \
        --name live-town --stride 10 --parallel 8 \
        > /tmp/hang.log 2>&1 < /dev/null &

行为：
  - **首次**运行（还没有存档）：不带走 --resume，按 --start 开新局；
    之后每次都带 --resume，从最近一份可用存档继续；
  - start.py **正常结束**（退出码 0：全灭 / 磁盘停止 / 步数用完）→ 监督结束，不再拉起；
  - start.py **崩溃**（退出码非 0，含被 OOM killer 杀掉）→ 等待若干秒后自动续跑；
    拉起前会先确认 LLM 服务可用（否则等它起来，避免空转重启）；
  - 连续崩溃超过 --max-restarts 次即停止（避免死循环刷日志）；
  - 单次运行超过 --reset-uptime 秒视为「健康运行」，崩溃计数归零（偶发崩溃不算连败）；
  - 磁盘剩余低于 --min-free-gb 时不再拉起（start.py 内部每步也会自查）。
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

CHILD = None          # 当前子进程
STOPPING = False      # 收到停止信号


def log(msg):
    print("[forever {}] {}".format(datetime.now().strftime("%m-%d %H:%M:%S"), msg), flush=True)


def disk_free_gb(path):
    probe = path
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    try:
        return shutil.disk_usage(probe or ".").free / (1024 ** 3)
    except OSError:
        return float("inf")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def llm_ready(url, timeout=5.0):
    """LLM 服务是否可用（Ollama 挂了的话，拉起 start.py 只会立刻再崩）。"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def wait_for_llm(url, max_wait=1800.0, interval=20.0):
    """等 LLM 服务恢复；超时返回 False。"""
    waited = 0.0
    while waited < max_wait:
        if STOPPING:
            return False
        if llm_ready(url):
            return True
        if waited == 0.0:
            log("LLM 服务不可用，等待它恢复（最多 {:.0f} 分钟）…".format(max_wait / 60))
        time.sleep(interval)
        waited += interval
    return False


def on_signal(signum, frame):
    """收到停止信号：转发给子进程（start.py 会写完当前步再退），然后退出循环。"""
    global STOPPING
    STOPPING = True
    log("收到信号 {}，准备停止…".format(signum))
    if CHILD and CHILD.poll() is None:
        CHILD.send_signal(signum)


def main():
    global CHILD
    ap = argparse.ArgumentParser(description="挂机监督：崩溃自动续跑（start.py --forever）")
    ap.add_argument("--name", type=str, required=True, help="模拟名称（同 start.py --name）")
    ap.add_argument("--python", type=str, default=sys.executable, help="运行 start.py 的解释器")
    ap.add_argument("--script", type=str, default="start.py", help="被监督的脚本（默认 start.py）")
    ap.add_argument("--start", type=str, default="20250214-09:30", help="首次开局的起始模拟时间")
    ap.add_argument("--stride", type=int, default=10, help="每步跨多少模拟分钟")
    ap.add_argument("--parallel", type=int, default=8, help="阶段化并行线程数")
    ap.add_argument("--verbose", type=str, default="info", help="日志级别")
    ap.add_argument("--min-free-gb", type=float, default=5.0, help="磁盘下限（GB）")
    ap.add_argument("--max-restarts", type=int, default=20, help="连续崩溃上限")
    ap.add_argument("--restart-delay", type=float, default=15.0, help="崩溃后等待秒数")
    ap.add_argument("--reset-uptime", type=float, default=600.0, help="单次运行超过该秒数即重置崩溃计数")
    ap.add_argument("--llm-url", type=str, default="http://127.0.0.1:11434/api/version",
                    help="LLM 服务探活地址（留空则跳过检查）")
    args = ap.parse_args()

    # start.py 里全是相对路径，必须在 generative_agents 目录下运行
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    ckpt_dir = os.path.join("results", "checkpoints", args.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 单实例锁：避免同一存档被两个监督器同时推进
    lock_path = os.path.join(ckpt_dir, ".forever.pid")
    if os.path.exists(lock_path):
        try:
            old_pid = int(open(lock_path, encoding="utf-8").read().strip())
        except (ValueError, OSError):
            old_pid = 0
        if old_pid and old_pid != os.getpid() and pid_alive(old_pid):
            log("已有监督器在跑（pid {}），退出。如确认它已死，删掉 {} 再试。".format(old_pid, lock_path))
            return 1
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    log("挂机开始：name={} stride={} parallel={} 磁盘下限={}GB".format(
        args.name, args.stride, args.parallel, args.min_free_gb))

    failures = 0
    rounds = 0
    while not STOPPING:
        # 磁盘看门狗
        free = disk_free_gb(ckpt_dir)
        if free < args.min_free_gb:
            log("磁盘剩余 {:.1f}GB < {:.1f}GB，停止挂机（存档已保留）".format(free, args.min_free_gb))
            break

        # LLM 探活（避免空转重启）
        if args.llm_url and not wait_for_llm(args.llm_url):
            log("等待 LLM 服务超时，停止挂机")
            break

        # 是否已有存档决定用 resume 还是开局
        has_ckpt = any(
            f.startswith("simulate-") and f.endswith(".json")
            for f in os.listdir(ckpt_dir)
        )
        cmd = [args.python, args.script, "--name", args.name, "--forever",
               "--stride", str(args.stride), "--parallel", str(args.parallel),
               "--verbose", args.verbose, "--min-free-gb", str(args.min_free_gb)]
        if has_ckpt:
            cmd.append("--resume")
        else:
            cmd += ["--start", args.start]

        rounds += 1
        log("第 {} 次拉起（{}）： {}".format(rounds, "续跑" if has_ckpt else "开新局", " ".join(cmd[1:])))
        started = time.time()
        CHILD = subprocess.Popen(cmd, cwd=here)
        rc = CHILD.wait()
        CHILD = None
        uptime = time.time() - started

        if STOPPING:
            log("已按要求停止（子进程退出码 {}）".format(rc))
            break

        if rc == 0:
            log("start.py 正常结束（退出码 0），挂机收工。共拉起 {} 次。".format(rounds))
            break

        # 崩溃处理
        if uptime >= args.reset_uptime:
            log("上次运行 {:.0f} 秒（>= {:.0f} 秒），视为健康运行，崩溃计数归零".format(uptime, args.reset_uptime))
            failures = 0

        failures += 1
        log("start.py 异常退出（退出码 {}，运行 {:.0f} 秒），连续第 {} 次".format(rc, uptime, failures))
        if failures > args.max_restarts:
            log("连续崩溃超过 {} 次，停止挂机（建议查看日志后再手动续跑）".format(args.max_restarts))
            break

        # 等一会儿再拉起（避免瞬时故障立刻重试）
        for _ in range(int(args.restart_delay)):
            if STOPPING:
                break
            time.sleep(1)

    try:
        os.remove(lock_path)
    except OSError:
        pass
    log("监督器退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
