import os
import sys
import json
import time
import argparse
import threading
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify

# compress.py 与 start.py 都在模块级直接执行 argparse.parse_args()，
# 为避免 replay.py 自己的启动参数被它们误解析，import 之前先把 argv 收干净。
_argv_backup = sys.argv
sys.argv = sys.argv[:1]
from compress import frames_per_step, file_movement, life_events_file, MovementBuilder
sys.argv = _argv_backup


app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static",
    static_url_path="/static",
)

checkpoints_root = "results/checkpoints"
compressed_root = "results/compressed"

# 步耗时兜底值：模拟推进一个 step 所需真实毫秒数（本机实测约 5.5 分钟）
default_step_ms = 330 * 1000
# 步耗时的合理区间，用于夹住由文件 mtime 推算出的异常结果
min_step_ms = 10 * 1000
max_step_ms = 3600 * 1000
# 判定「模拟已结束」的宽松系数：空闲时间超过 该系数 × 平均步耗时 才认为结束
finish_grace = 3

# ---------- 直播台（broadcast）参数 ----------
# 固定纪元：轮内相位是「当前时间的纯函数」，所以服务重启、换容器、重新部署之后，
# 所有观众算出来的位置都自动一致，不需要任何持久化状态。
broadcast_anchor_ms = 1767225600000  # 2026-01-01T00:00:00Z
# 每个 step 占用的真实毫秒数：1 分钟/步，即模拟时间以 10 倍速前进
broadcast_ms_per_step = 60 * 1000

# ---------- 直播台的全场虚拟时钟（主播全局控制：暂停/倍率对所有人生效）----------
# 只存最小状态；任何操作都先重算基准再改状态，保证暂停/恢复/变速都不会引起画面跳变。
# 重启即回到直播边：倍率回 1x、取消暂停，相位重新由墙钟决定（与无按钮时一致）。
_bc_lock = threading.RLock()
_bc = {"paused": False, "rate": 1.0, "vbase": 0.0, "wbase": 0.0}


def _bc_init():
    """初始化：虚拟时刻 == 墙钟时刻（1x、不暂停）"""
    now = time.time() * 1000
    _bc["vbase"] = now
    _bc["wbase"] = now


def _bc_virtual(now_ms=None):
    """当前的全场虚拟时刻（毫秒，与 broadcast_anchor_ms 同一坐标系）"""
    with _bc_lock:
        if _bc["paused"]:
            return _bc["vbase"]
        if now_ms is None:
            now_ms = time.time() * 1000
        return _bc["vbase"] + (now_ms - _bc["wbase"]) * _bc["rate"]


def _bc_rebase(now_ms):
    """把基准挪到当前虚拟时刻；之后的改状态操作就不会让画面跳变"""
    _bc["vbase"] = _bc_virtual(now_ms)
    _bc["wbase"] = now_ms


def _bc_snapshot():
    return {"virtual_ms": _bc_virtual(), "paused": _bc["paused"], "rate": _bc["rate"]}


def bc_pause():
    with _bc_lock:
        _bc_rebase(time.time() * 1000)
        _bc["paused"] = True
        return _bc_snapshot()


def bc_resume():
    with _bc_lock:
        _bc_rebase(time.time() * 1000)
        _bc["paused"] = False
        return _bc_snapshot()


def bc_set_rate(rate):
    with _bc_lock:
        _bc_rebase(time.time() * 1000)
        _bc["rate"] = rate
        return _bc_snapshot()


# 模拟刚启动、还没产出第一步时展示的等待页（会自动重试）
waiting_page = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>小镇正在启动</title>
  <meta http-equiv="refresh" content="10">
</head>
<body style="font-family: sans-serif; padding: 3em; text-align: center; color: #333;">
  <h2>『{name}』正在初始化</h2>
  <p>第一步需要为全体居民生成完整日程，大约几分钟。</p>
  <p>本页每 10 秒会自动重试，无需手动刷新。</p>
</body>
</html>"""


def has_checkpoints(name):
    """该模拟是否具备实时推进的数据源（存在 simulate-*.json 存档）"""
    folder = os.path.join(checkpoints_root, name)
    if not os.path.isdir(folder):
        return False
    for file_name in os.listdir(folder):
        if file_name.startswith("simulate-") and file_name.endswith(".json"):
            return True
    return False


class LiveSession:
    """一次实时模拟的增量转换器与会话状态。

    由前端轮询驱动：每次 refresh() 扫描存档目录，把新出现的存档喂给
    MovementBuilder，产出的帧累积在内存中，同时落盘到
    compressed/<name>/movement.json —— 即「边推边攒」，
    于是模拟结束后无需再跑 compress.py 就已经有了完整的回放数据。
    """

    def __init__(self, name):
        self.name = name
        self.checkpoints_folder = os.path.join(checkpoints_root, name)
        self.compressed_folder = os.path.join(compressed_root, name)
        self.builder = MovementBuilder()
        self.processed = []      # 已处理的存档文件名（有序）
        self.file_times = []     # 与 processed 一一对应的 mtime
        self.conversation = {}
        self.last_new_data = time.time()
        self._lock = threading.Lock()

    # ---------- 增量转换 ----------
    def _list_checkpoints(self):
        if not os.path.isdir(self.checkpoints_folder):
            return []
        files = []
        for file_name in os.listdir(self.checkpoints_folder):
            if file_name.startswith("simulate-") and file_name.endswith(".json"):
                files.append(file_name)
        return sorted(files)

    def refresh(self):
        """扫描并处理新存档，返回是否吃到新数据"""
        with self._lock:
            new_files = [f for f in self._list_checkpoints() if f not in self.processed]
            if len(new_files) < 1:
                return False

            # 对话是逐步累积的，每次取最新一份
            conv_path = os.path.join(self.checkpoints_folder, "conversation.json")
            if os.path.exists(conv_path):
                try:
                    with open(conv_path, "r", encoding="utf-8") as f:
                        self.conversation = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass  # 模拟进程可能正在写入，下一轮再取

            # 生命事件（死亡/出生）：也逐步累积，用于花名册与墓碑
            life_path = os.path.join(self.checkpoints_folder, life_events_file)
            if os.path.exists(life_path):
                try:
                    with open(life_path, "r", encoding="utf-8") as f:
                        self.builder.add_life_events(json.load(f))
                except (json.JSONDecodeError, OSError):
                    pass

            for file_name in new_files:
                path = os.path.join(self.checkpoints_folder, file_name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue  # 文件尚未写完，留到下一轮
                self.builder.add_step(json_data, self.conversation)
                self.processed.append(file_name)
                self.file_times.append(os.path.getmtime(path))

            # 用文件自身的写入时间，而非当前时间：否则服务刚启动时，
            # 会把几小时前就跑完的模拟误判成「正在运行」
            if len(self.file_times) > 0:
                self.last_new_data = self.file_times[-1]
            self._dump()
            return True

    def _dump(self):
        """边推边攒：把累积结果落盘，任何时刻都能用静态方式回放"""
        try:
            os.makedirs(self.compressed_folder, exist_ok=True)
            with open(os.path.join(self.compressed_folder, file_movement), "w", encoding="utf-8") as f:
                f.write(json.dumps(self.builder.result(), indent=2, ensure_ascii=False))
        except OSError:
            pass

    # ---------- 状态 ----------
    @property
    def avg_step_ms(self):
        """已完成步骤的平均真实耗时（毫秒）。

        结果会被夹在 [min_step_ms, max_step_ms] 内：若存档是被批量拷贝进来的
        而非逐步产生，文件 mtime 会挤在一起，直接换算会得到近乎为零的节奏，
        进而让前端把人物移动速度放大到荒谬的程度。
        """
        if len(self.file_times) < 2:
            return default_step_ms
        span = self.file_times[-1] - self.file_times[0]
        avg = span / (len(self.file_times) - 1) * 1000
        return int(min(max(avg, min_step_ms), max_step_ms))

    @property
    def status(self):
        if len(self.processed) < 1:
            return "waiting"
        idle = time.time() - self.last_new_data
        if idle > max(600, self.avg_step_ms / 1000.0 * finish_grace):
            return "finished"
        return "running"

    @property
    def latest_frame(self):
        latest = 0
        for key in self.builder.all_movement.keys():
            if key in ("description", "conversation"):
                continue
            try:
                latest = max(latest, int(key))
            except ValueError:
                continue
        return latest

    # ---------- 数据导出 ----------
    def frames_since(self, since):
        """返回帧号大于 since 的帧（增量接口用）"""
        frames = {}
        for key, value in self.builder.all_movement.items():
            if key in ("description", "conversation"):
                continue
            try:
                frame_no = int(key)
            except ValueError:
                continue
            if frame_no > since:
                frames[key] = value
        return {
            "frames": frames,
            "conversation": self.builder.all_movement["conversation"],
            "description": self.builder.all_movement["description"],
            "roster": self.builder.roster(),
            "latest_frame": self.latest_frame,
            "avg_step_ms": self.avg_step_ms,
            "status": self.status,
        }

    def spawn_positions(self, target_step):
        """首屏人物的落点：取该步内每个角色的第一个已知坐标，避免「从旧位置走过去」。

        所有「存在过」的角色都会给一个落点（含中途出生、入学、已故者）：
        前端先按名单建好精灵，再由 apply_roster 按帧号决定显隐——已故者因此
        能正确地「在首屏就是墓碑」或「随播放进度变为墓碑」。
        """
        start = (target_step - 1) * frames_per_step + 1
        end = target_step * frames_per_step
        positions = {}
        for agent, entry in self.builder.roster().items():
            coord = None
            for frame_no in range(start, end + 1):
                frame = self.builder.all_movement.get(str(frame_no))
                if frame and agent in frame:
                    coord = frame[agent]["movement"]
                    break
            if coord is None:
                frame0 = self.builder.all_movement.get("0", {})
                if agent in frame0:
                    coord = frame0[agent]["movement"]
            if coord is None:
                coord = self.builder.first_seen_coord(agent) or entry.get("coord")
            if coord is not None:
                positions[agent] = coord
        return positions

    def build_payload(self, target_step=0):
        """构造首屏注入数据。target_step 为 0 表示跳到最新一步。"""
        step_count = self.builder.step
        if step_count < 1:
            return None
        if target_step < 1 or target_step > step_count:
            target_step = step_count

        start = (target_step - 1) * frames_per_step + 1
        end = target_step * frames_per_step

        frames = {}
        for frame_no in range(start, end + 1):
            key = str(frame_no)
            if key in self.builder.all_movement:
                frames[key] = self.builder.all_movement[key]

        start_datetime = ""
        if len(self.builder.start_datetime) > 0:
            t = datetime.fromisoformat(self.builder.start_datetime)
            t = t + timedelta(minutes=self.builder.stride * (target_step - 1))
            start_datetime = t.isoformat()

        return {
            "start_datetime": start_datetime,
            "stride": self.builder.stride,
            "sec_per_step": self.builder.stride,
            "persona_init_pos": self.spawn_positions(target_step),
            "roster": self.builder.roster(),
            "all_movement": dict(
                frames,
                description=self.builder.all_movement["description"],
                conversation=self.builder.all_movement["conversation"],
            ),
            "step": start,
            "latest_frame": self.latest_frame,
            "avg_step_ms": self.avg_step_ms,
            "status": self.status,
        }


_sessions = {}
_sessions_lock = threading.Lock()


def get_session(name):
    with _sessions_lock:
        if name not in _sessions:
            _sessions[name] = LiveSession(name)
        return _sessions[name]


def load_recording(name):
    """读取一份完整的录像（compress.py 的产物）；不存在则返回 None"""
    path = os.path.join(compressed_root, name, file_movement)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sidebar_names(params):
    """侧边栏花名册：列出所有存在过的角色（含已故）；缺 roster 的老录像退回初始位置名单。"""
    roster = params.get("roster") or {}
    if len(roster) > 0:
        return list(roster.keys())
    return list(params["persona_init_pos"].keys())


def broadcast_meta(params):
    """直播台的时间轴参数：帧号完全由「当前时间」决定，客户端不保存任何进度"""
    frames = [int(k) for k in params["all_movement"] if k not in ("description", "conversation")]
    total_frames = max(frames) if len(frames) > 0 else 0
    return {
        "total_frames": total_frames,
        "total_steps": (total_frames + frames_per_step - 1) // frames_per_step,
        "ms_per_frame": broadcast_ms_per_step // frames_per_step,
    }


_bc_init()   # 模块加载时初始化全场虚拟时钟（虚拟时刻 == 墙钟时刻）


@app.route("/now", methods=['GET'])
def now():
    """直播台客户端对齐全场时间轴用：返回服务端虚拟时刻与全场播放状态"""
    return jsonify(dict(_bc_snapshot(), server_now_ms=int(time.time() * 1000)))


@app.route("/bc/pause", methods=['POST', 'GET'])
def bc_pause_route():
    """全场暂停（对所有观众生效）"""
    return jsonify(bc_pause())


@app.route("/bc/resume", methods=['POST', 'GET'])
def bc_resume_route():
    """全场恢复播放"""
    return jsonify(bc_resume())


@app.route("/bc/rate", methods=['POST', 'GET'])
def bc_rate_route():
    """全场变速（相位保持，不跳帧）。用法：/bc/rate?r=2"""
    try:
        rate = float(request.args.get("r", 1))
    except (TypeError, ValueError):
        return jsonify({"error": "rate 必须是数字，例如 /bc/rate?r=2"}), 400
    if rate <= 0 or rate > 10:
        return jsonify({"error": "rate 超出范围（0 < r <= 10）"}), 400
    return jsonify(bc_set_rate(rate))


@app.route("/", methods=['GET'])
def index():
    name = request.args.get("name", "")          # 记录名称
    zoom = float(request.args.get("zoom", 0.8))  # 画面缩放比例（纯本地视觉，不进时间轴）
    mode = request.args.get("mode", "").lower()  # 留空 = 直播台；static / live 是隐藏入口

    if len(name) < 1:
        return f"Invalid name of the simulation: '{name}'"

    if mode not in ("", "broadcast", "static", "live"):
        return (f"Invalid mode '{mode}': 留空或 mode=broadcast 是直播台，"
                f"mode=static 是录播，mode=live 是实时。")

    # ---------- 直播台（默认入口）----------
    # 画面内容完全由「当前时间 + 固定纪元」推出，所以两个观众看到的是同一时刻的小镇。
    # 时间轴参数一律收在服务端常量里：URL 上的 step / speed / k 在这里被彻底忽略。
    if mode in ("", "broadcast"):
        params = load_recording(name)
        if params is None:
            return (f"'{name}' 还没有录像文件 {compressed_root}/{name}/{file_movement}，无法开播。"
                    f"<br />先跑：python compress.py --name {name}")
        return render_template(
            "index.html",
            persona_names=sidebar_names(params),
            step=1,
            play_speed=2 ** 2,
            zoom=zoom,
            is_live=False,
            is_broadcast=True,
            sim_name=name,
            k=0.0,
            k_explicit=False,
            avg_step_ms=0,
            latest_frame=0,
            run_status="broadcast",
            frames_per_step=frames_per_step,
            anchor_ms=broadcast_anchor_ms,
            server_now_ms=int(time.time() * 1000),
            **broadcast_meta(params),
            **params,
        )

    # ---------- 隐藏入口：录播 / 实时 ----------
    step = int(request.args.get("step", 0))      # 回放起始步数（仅隐藏入口使用）
    speed = int(request.args.get("speed", 2))    # 回放速度（0~5，静态模式生效）
    k_arg = request.args.get("k", "")            # 倍率（实时模式下生效）

    has_ckpt_folder = os.path.isdir(os.path.join(checkpoints_root, name))

    # 传了 mode=static / mode=live 时以显式指定的为准 —— 这样已经跑过的模拟（目录里
    # 留着 checkpoints）也能用 &mode=static 当录播看，不必把存档目录搬走。
    if mode == "static":
        live = False
    else:
        if not has_ckpt_folder:
            return (f"'{name}' 没有存档目录 results/checkpoints/{name}，无法实时播放。<br />"
                    f"想看录播请改用 mode=static，前提是已经跑过 compress.py --name {name}。")
        live = True

    # 倍率：实时模式默认 2；静态模式默认沿用 speed，传了 k 才切到倍率控制
    k_explicit = len(k_arg) > 0
    try:
        k = float(k_arg) if k_explicit else (2.0 if live else 0.0)
    except ValueError:
        k_explicit = False
        k = 2.0 if live else 0.0

    if live:
        session = get_session(name)
        session.refresh()
        params = session.build_payload(step)
        if params is None:
            # 目录已建但第一步还没落盘，给一个会自动重试的等待页
            return waiting_page.format(name=name)

        return render_template(
            "index.html",
            persona_names=sidebar_names(params),
            step=params["step"],
            play_speed=2 ** speed,
            zoom=zoom,
            is_live=True,
            is_broadcast=False,
            sim_name=name,
            k=k,
            k_explicit=k_explicit,
            avg_step_ms=params["avg_step_ms"],
            latest_frame=params["latest_frame"],
            run_status=params["status"],
            frames_per_step=frames_per_step,
            start_datetime=params["start_datetime"],
            stride=params["stride"],
            sec_per_step=params["sec_per_step"],
            persona_init_pos=params["persona_init_pos"],
            roster=params.get("roster", {}),
            all_movement=params["all_movement"],
        )

    # ---------- 静态回放 ----------
    replay_file = f"{compressed_root}/{name}/{file_movement}"
    if not os.path.exists(replay_file):
        return f"The data file doesn‘t exist: '{replay_file}'<br />Run compress.py to generate the data first."

    with open(replay_file, "r", encoding="utf-8") as f:
        params = json.load(f)

    if step < 1:
        step = 1
    if step > 1:
        # 重新设置回放的起始时间
        t = datetime.fromisoformat(params["start_datetime"])
        dt = t + timedelta(minutes=params["stride"]*(step-1))
        params["start_datetime"] = dt.isoformat()
        step = (step-1) * frames_per_step + 1
        if step >= len(params["all_movement"]):
            step = len(params["all_movement"])-1

        # 重新设置Agent的初始位置：该步在场上的人用当前坐标，
        # 其余（未出生 / 已故 / 婴幼儿）沿用花名册里的坐标——
        # 前端会按帧号决定显隐，所以名单不能剔除。
        persona_init_pos = params["persona_init_pos"]
        persona_step_pos = params["all_movement"][f"{step}"]
        roster = params.get("roster") or {}
        for agent in list(persona_init_pos.keys()):
            if agent in persona_step_pos:
                persona_init_pos[agent] = persona_step_pos[agent]["movement"]
            elif roster.get(agent, {}).get("coord"):
                persona_init_pos[agent] = roster[agent]["coord"]

    if speed < 0:
        speed = 0
    elif speed > 5:
        speed = 5
    speed = 2 ** speed

    return render_template(
        "index.html",
        persona_names=sidebar_names(params),
        step=step,
        play_speed=speed,
        zoom=zoom,
        is_live=False,
        is_broadcast=False,
        sim_name=name,
        k=k,
        k_explicit=k_explicit,
        avg_step_ms=0,
        latest_frame=0,
        run_status="static",
        frames_per_step=frames_per_step,
        **params
    )


@app.route("/frames", methods=['GET'])
def frames():
    """增量接口：返回帧号大于 since 的所有帧"""
    name = request.args.get("name", "")
    try:
        since = int(request.args.get("since", 0))
    except ValueError:
        since = 0

    if len(name) < 1 or not has_checkpoints(name):
        return jsonify({"error": "not a live simulation"}), 404

    session = get_session(name)
    session.refresh()
    return jsonify(session.frames_since(since))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="replay server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=6006, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="开启调试模式（勿对公网使用）")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
