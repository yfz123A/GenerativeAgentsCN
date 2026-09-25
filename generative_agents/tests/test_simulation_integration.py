"""L2 集成测试：fake LLM 下真实推进 SimulateServer.simulate() 主循环。

覆盖评审确定的六条不变量：
- S1 常规节奏：存档/对话落盘且 JSON 合法、角色集合不变、坐标在地图内
- S2 跨模拟日：生命循环确定性触发（全员发病 -> 婚配 -> 受孕 -> 分娩）
- S3a 断点续跑（step 连续）、S3b 坏存档回退、S3c 全灭优雅退出
"""

import json
from pathlib import Path

import pytest

from modules import life


# ---------- 工具 ----------

def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _archives(folder):
    return sorted(Path(folder).glob("simulate-*.json"))


def _latest_archive(folder):
    files = _archives(folder)
    assert files, "没有找到任何存档"
    return _read_json(files[-1])


# 强制确定性的档位：全员每日发病（轻症），无人死亡
UT_SICK_PROFILE = {
    "sick": [(999, 1.0)],
    "death": [(999, 0.0)],
    "severe_ratio": [(999, 0.0)],
    "severe_die": [(999, 0.0)],
    "mild_days": (2, 2),
    "severe_days": (3, 3),
}

# 强制确定性的档位：全员每日寿终
UT_DEATH_PROFILE = {
    "sick": [(999, 0.0)],
    "death": [(0, 1.0)],
    "severe_ratio": [(999, 0.0)],
    "severe_die": [(999, 0.0)],
    "mild_days": (1, 1),
    "severe_days": (2, 2),
}


@pytest.fixture
def sim_env(tmp_path, sim_name, static_agents_guard, fake_llm, start_module):
    """集成环境：存档写 tmp_path，唯一的仓库名 + 产物自动清理。"""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(parents=True)

    class Env:
        pass

    env = Env()
    env.folder = checkpoints / sim_name
    env.fake = fake_llm
    env.start_module = start_module

    def build(config=None, start_step=0, start_time="20250213-09:30", stride=15):
        if config is None:
            config = start_module.get_config(start_time, stride, life.seed_personas())
        server = start_module.SimulateServer(
            sim_name, "frontend/static", str(env.folder), config,
            start_step, "warn", "", parallel=1, min_free_gb=5.0,
        )
        # 预建记忆索引目录，保证 associate 存档可以落盘、续跑时可加载
        storage = Path("results/checkpoints") / sim_name / "storage"
        for agent in server.game.agents:
            (storage / agent / "associate").mkdir(parents=True, exist_ok=True)
        return server

    env.build = build
    return env


# ---------- S1 常规节奏 ----------

def test_s1_regular_steps_produce_valid_archives(sim_env):
    roster = set(life.seed_personas())
    server = sim_env.build()
    server.simulate(3, 15)

    assert server.exit_reason == "finished"
    assert len(_archives(sim_env.folder)) == 3  # 每步一个原子存档

    # 对话记录合法落盘
    conversation = _read_json(sim_env.folder / "conversation.json")
    assert isinstance(conversation, dict)

    # 最新存档：角色集合 == 初始花名册，坐标在地图内（100 高 x 140 宽）
    data = _latest_archive(sim_env.folder)
    assert set(data["agents"]) == roster
    assert data["step"] == 3
    for info in data["agents"].values():
        x, y = info["coord"]
        assert 0 <= x < 140 and 0 <= y < 100
        assert info["life"]["health"] == "healthy"  # 一天内无生命事件
    assert data["children"] == []
    assert data["housing_pool"] == [["the Ville", "周氏家族的房子", "空卧室"]]

    # 真实走到了 fake LLM 的决策链路
    assert sim_env.fake.agent_calls, "agent_think 应当产生 LLM 调用"
    hints = {hint for _, hint in sim_env.fake.agent_calls}
    assert {"wake_up", "schedule_daily", "schedule_decompose"} <= hints


# ---------- S2 跨模拟日（生命循环，强制确定性触发） ----------

def test_s2_life_day_crossing_triggers_life_events(sim_env, monkeypatch):
    roster = set(life.seed_personas())
    # 不碰运气：婚配必中、受孕骰子必过、全员每日轻症
    monkeypatch.setattr(life, "marriage_roll", lambda day_index, dice_seed: True)
    monkeypatch.setattr(
        life, "conception_roll", lambda husband, wife, day_index, dice_seed: 0.0
    )
    monkeypatch.setitem(life.PROFILES, "standard", UT_SICK_PROFILE)

    server = sim_env.build()
    server.simulate(5, 720)  # 5 步 x 12 小时 = 跨 2 个模拟日

    assert server.exit_reason == "finished"
    events = _read_json(sim_env.folder / "life_events.json")
    types = [e["type"] for e in events]

    # 第 1 模拟日：25 人全部发病 + 一场婚礼 + 新娘受孕
    # 第 2 模拟日：部分人转重（其余维持轻症）+ 第二场婚礼 + 分娩一个新生儿
    # 注：轻症 recover_day = 第 1 日 + 2 = 第 3 日，第 2 日不可能痊愈
    assert types.count("sick") == len(roster)
    assert types.count("marriage") == 2
    assert types.count("conceive") == 2
    assert types.count("birth") == 1
    assert types.count("recover") == 0
    assert types.count("worsen") <= len(roster)

    # 婚姻闭环：新房、配偶互写、婚礼叙事进入对话记录
    wedding = next(e for e in events if e["type"] == "marriage")
    groom, bride = wedding["name"], wedding["spouse"]
    data = _latest_archive(sim_env.folder)
    assert data["agents"][groom]["life"]["spouse"] == bride
    assert data["agents"][bride]["life"]["spouse"] == groom
    assert data["homes"][groom] == data["homes"][bride]
    assert data["housing_pool"] == []  # 住房池被首场婚礼取用

    conversation = json.dumps(
        _read_json(sim_env.folder / "conversation.json"), ensure_ascii=False
    )
    assert "结为夫妻" in conversation  # fake 婚礼叙事写入直播对话

    # 生育闭环：父母对应、幼儿登记、母亲产子状态复位
    birth = next(e for e in events if e["type"] == "birth")
    assert birth["parents"] == [groom, bride]
    assert birth["gender"] in ("男", "女")
    assert data["agents"][bride]["life"]["pregnant_by"] == ""
    assert data["agents"][bride]["life"]["births"] == 1
    assert len(data["children"]) == 1
    assert data["children"][0]["name"] == birth["name"]

    # 生命状态随存档持久化（续跑不重置病情/婚姻）
    assert data["agents"][bride]["life"]["health"] in ("mild", "severe")


# ---------- S3a 断点续跑 ----------

def test_s3a_resume_from_checkpoint_continues_step(sim_env):
    roster = set(life.seed_personas())
    server = sim_env.build()
    server.simulate(2, 15)
    assert server.last_step == 2

    # 从存档恢复（与 start.py --resume 相同的路径）
    sim_cfg = sim_env.start_module.get_config_from_log(str(sim_env.folder))
    assert sim_cfg is not None
    assert sim_cfg["step"] == 2
    assert sim_cfg["time"]["start"] == "20250213-10:00"  # 09:45 + 15 分钟

    server2 = sim_env.build(config=sim_cfg, start_step=sim_cfg["step"])
    server2.simulate(1, 15)

    assert server2.exit_reason == "finished"
    assert server2.last_step == 3
    data = _latest_archive(sim_env.folder)
    assert data["step"] == 3
    assert set(data["agents"]) == roster  # 续跑后角色集合不变

    # 复活的生命原点：续跑后生命时钟连续（不会重掷已判定的日子）
    assert data.get("life_origin") == "20250213-09:30"


# ---------- S3b 坏存档回退 ----------

def test_s3b_corrupt_archive_falls_back_to_previous(sim_env):
    server = sim_env.build()
    server.simulate(2, 15)

    # 三个坏档：半个 JSON / 缺字段 / 时间格式非法（字典序都比好档新）
    (sim_env.folder / "simulate-99999999-0000.json").write_text(
        '{"time": "2025', encoding="utf-8"
    )
    (sim_env.folder / "simulate-99999998-0000.json").write_text('{"foo": 1}', encoding="utf-8")
    (sim_env.folder / "simulate-99999997-0000.json").write_text(
        json.dumps({"time": "garbage", "step": 9, "agents": {}}), encoding="utf-8"
    )

    cfg = sim_env.start_module.get_config_from_log(str(sim_env.folder))
    assert cfg is not None
    assert cfg["step"] == 2  # 回退到最新的好档

    # 回退后的配置真的能续跑
    server2 = sim_env.build(config=cfg, start_step=cfg["step"])
    server2.simulate(1, 15)
    assert server2.last_step == 3


# ---------- S3c 全灭优雅退出 ----------

def test_s3c_extinct_population_exits_gracefully(sim_env, monkeypatch):
    roster = set(life.seed_personas())
    monkeypatch.setitem(life.PROFILES, "standard", UT_DEATH_PROFILE)

    server = sim_env.build()
    server.simulate(5, 720)  # 第 1 个跨日 tick 全员寿终

    assert server.exit_reason == "extinct"
    assert server.agent_status == {}

    events = _read_json(sim_env.folder / "life_events.json")
    deaths = [e for e in events if e["type"] == "death"]
    assert {e["name"] for e in deaths} == roster
    assert all(e["cause"] == "寿终" for e in deaths)

    # 全灭发生在第 3 步的 life_tick：第 1、2 步的存档完整保留
    assert len(_archives(sim_env.folder)) == 2
