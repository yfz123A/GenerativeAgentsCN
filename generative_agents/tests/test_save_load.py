"""L1 单元测试：start.py 的存档工具函数（原子写 / 坏档回退 / 续跑自检）。"""

import json
import os

import pytest


# ============ atomic_write_json ============

def test_atomic_write_creates_valid_json(tmp_path, start_module):
    path = tmp_path / "存档.json"
    start_module.atomic_write_json(str(path), {"名字": "值", "step": 1})
    text = path.read_text(encoding="utf-8")
    assert "名字" in text  # ensure_ascii=False，中文原样落盘
    assert json.loads(text) == {"名字": "值", "step": 1}
    assert not os.path.exists(str(path) + ".tmp")  # 没有残留临时文件


def test_atomic_write_overwrites_completely(tmp_path, start_module):
    path = tmp_path / "存档.json"
    start_module.atomic_write_json(str(path), {"step": 1, "old": True})
    start_module.atomic_write_json(str(path), {"step": 2})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"step": 2}


# ============ get_config_from_log ============

GOOD_ARCHIVE = {
    "time": "20250213-10:00",
    "step": 5,
    "stride": 15,
    "agents": {"陈守信": {"coord": [1, 2]}},
}


def _write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_get_config_from_latest_good_archive(tmp_path, start_module):
    _write(tmp_path / "simulate-20250213-1000.json", GOOD_ARCHIVE)
    (tmp_path / "conversation.json").write_text("{}", encoding="utf-8")
    (tmp_path / "life_events.json").write_text("[]", encoding="utf-8")

    cfg = start_module.get_config_from_log(str(tmp_path))
    assert cfg["step"] == 5
    # 恢复时刻 = 存档时间 + stride
    assert cfg["time"]["start"] == "20250213-10:15"
    # config_path 被重写到静态设定
    assert cfg["agents"]["陈守信"]["config_path"].endswith(
        "assets/village/agents/陈守信/agent.json"
    )


def test_corrupt_newest_archive_is_skipped(tmp_path, start_module):
    _write(tmp_path / "simulate-20250213-1000.json", GOOD_ARCHIVE)
    # 崩溃留下的半个 JSON（字典序最新，优先被尝试）
    (tmp_path / "simulate-20250213-1015.json").write_text(
        '{"time": "2025', encoding="utf-8"
    )
    cfg = start_module.get_config_from_log(str(tmp_path))
    assert cfg is not None
    assert cfg["step"] == 5  # 回退到上一个好档


def test_archive_missing_fields_is_skipped(tmp_path, start_module):
    _write(tmp_path / "simulate-20250213-1000.json", GOOD_ARCHIVE)
    _write(tmp_path / "simulate-20250213-1015.json", {"foo": 1})  # 缺 agents/time
    cfg = start_module.get_config_from_log(str(tmp_path))
    assert cfg["step"] == 5


def test_archive_bad_time_format_is_skipped(tmp_path, start_module):
    _write(tmp_path / "simulate-20250213-1000.json", GOOD_ARCHIVE)
    _write(
        tmp_path / "simulate-20250213-1015.json",
        {"time": "garbage", "step": 9, "agents": {}},
    )
    cfg = start_module.get_config_from_log(str(tmp_path))
    assert cfg["step"] == 5


def test_special_files_are_excluded(tmp_path, start_module):
    """conversation.json / life_events.json 不是存档，即使内容像存档也不能选。"""
    _write(tmp_path / "simulate-20250213-1000.json", GOOD_ARCHIVE)
    _write(tmp_path / "life_events.json", {"time": "20250213-23:00", "step": 99, "agents": {}})
    _write(tmp_path / "conversation.json", {"time": "20250213-23:00", "step": 99, "agents": {}})
    cfg = start_module.get_config_from_log(str(tmp_path))
    assert cfg["step"] == 5


def test_no_usable_archive_returns_none(tmp_path, start_module):
    (tmp_path / "simulate-20250213-1000.json").write_text("{bad", encoding="utf-8")
    assert start_module.get_config_from_log(str(tmp_path)) is None


def test_empty_folder_returns_none(tmp_path, start_module):
    assert start_module.get_config_from_log(str(tmp_path)) is None


# ============ missing_agents ============

def test_missing_agents_detects_renamed_agent(tmp_path, start_module):
    agent_dir = tmp_path / "assets/village/agents/陈守信"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.json").write_text("{}", encoding="utf-8")

    config = {
        "agents": {
            "陈守信": {"config_path": "assets/village/agents/陈守信/agent.json"},
            "王小明": {},  # config_path 缺省 -> 按默认路径推导，文件不存在
        }
    }
    assert start_module.missing_agents(config, static_root=str(tmp_path)) == ["王小明"]


def test_missing_agents_all_present(tmp_path, start_module):
    agent_dir = tmp_path / "assets/village/agents/陈守信"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.json").write_text("{}", encoding="utf-8")

    config = {
        "agents": {
            "陈守信": {"config_path": "assets/village/agents/陈守信/agent.json"}
        }
    }
    assert start_module.missing_agents(config, static_root=str(tmp_path)) == []


def test_missing_agents_default_path_uses_underscore(tmp_path, start_module):
    """默认 config_path 用 name.replace(' ', '_') 推导。"""
    agent_dir = tmp_path / "assets/village/agents/Chen_Shouxin"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.json").write_text("{}", encoding="utf-8")

    config = {"agents": {"Chen Shouxin": {}}}
    assert start_module.missing_agents(config, static_root=str(tmp_path)) == []
