"""L1 单元测试：modules/life.py 生命骰子引擎。

设计原则（与产品代码注释一致）：全部随机数由 (dice_seed, 角色名, 模拟日)
派生 —— 本文件验证确定性与状态机行为，不验证统计分布。
"""

import copy
import datetime

import pytest

from modules import life


# ============ 时间纯函数 ============

def test_parse_sim_time():
    t = life.parse_sim_time("20250213-09:30")
    assert t == datetime.datetime(2025, 2, 13, 9, 30)


def test_sim_days_elapsed():
    start = datetime.datetime(2025, 2, 13, 9, 30)
    assert life.sim_days_elapsed(start, start) == 0.0
    assert life.sim_days_elapsed(start + datetime.timedelta(days=1), start) == 1.0
    assert life.sim_days_elapsed(start + datetime.timedelta(hours=12), start) == 0.5


def test_abstract_age_is_pure_function_of_sim_time():
    start = datetime.datetime(2025, 2, 13, 9, 30)
    # 1 模拟日 = 4 岁
    assert life.abstract_age(20, start, start) == 20.0
    assert life.abstract_age(20, start + datetime.timedelta(days=1), start) == 24.0
    assert life.abstract_age(20, start + datetime.timedelta(hours=12), start) == 22.0


# ============ 分段表与骰子 ============

def test_band_boundaries():
    table = [(18, 0.1), (40, 0.2), (999, 0.3)]
    assert life._band(table, 17.9) == 0.1
    assert life._band(table, 18.0) == 0.2   # 恰好压线取下一档
    assert life._band(table, 39.9) == 0.2
    assert life._band(table, 40.0) == 0.3
    assert life._band(table, 200.0) == 0.3  # 超出取最后一档


def test_rng_same_inputs_same_output():
    a = [life._rng(7, "甲", 3).random() for _ in range(5)]
    b = [life._rng(7, "甲", 3).random() for _ in range(5)]
    assert a == b


def test_rng_differs_across_days():
    a = [life._rng(7, "甲", 3).random() for _ in range(5)]
    b = [life._rng(7, "甲", 4).random() for _ in range(5)]
    assert a != b


# ============ daily_roll 状态机 ============

# 全零档位：除硬编码分支外一切事件概率为 0，用于隔离单个状态转移
UT_ZERO = {
    "sick": [(999, 0.0)],
    "death": [(999, 0.0)],
    "severe_ratio": [(999, 0.0)],
    "severe_die": [(999, 0.0)],
    "mild_days": (1, 1),
    "severe_days": (2, 2),
}

UT_SICK = {
    "sick": [(999, 1.0)],
    "death": [(999, 0.0)],
    "severe_ratio": [(999, 0.0)],
    "severe_die": [(999, 0.0)],
    "mild_days": (2, 2),
    "severe_days": (3, 3),
}

UT_SICK_SEVERE = dict(UT_SICK, severe_ratio=[(999, 1.0)])

UT_DEATH = {
    "sick": [(999, 0.0)],
    "death": [(0, 1.0)],
    "severe_ratio": [(999, 0.0)],
    "severe_die": [(999, 0.0)],
    "mild_days": (1, 1),
    "severe_days": (2, 2),
}


@pytest.fixture
def ut_zero(monkeypatch):
    monkeypatch.setitem(life.PROFILES, "ut_zero", UT_ZERO)
    return "ut_zero"


@pytest.fixture
def ut_sick(monkeypatch):
    monkeypatch.setitem(life.PROFILES, "ut_sick", UT_SICK)
    return "ut_sick"


def _state(health, since_day=0.0, recover_day=0.0):
    st = life.new_life_state()
    st["health"] = health
    st["since_day"] = since_day
    st["recover_day"] = recover_day
    return st


def _dice_seed_where(predicate, name="探针", day=7, limit=1000):
    """在确定性骰子里找一个满足条件的种子（替代碰运气）。"""
    for s in range(limit):
        if predicate(s):
            return s
    raise AssertionError("没有找到满足条件的骰子种子")


def test_healthy_no_event(ut_zero):
    st = _state(life.HEALTHY)
    new, event = life.daily_roll("张三", 30, 10, st, ut_zero, dice_seed=1)
    assert event is None
    assert new["health"] == life.HEALTHY


def test_sick_mild(ut_sick):
    st = _state(life.HEALTHY)
    new, event = life.daily_roll("张三", 30, 10, st, ut_sick, dice_seed=1)
    assert event == {"type": "sick", "severity": "mild"}
    assert new["health"] == life.MILD
    assert new["sick_count"] == 1
    assert new["since_day"] == 10
    assert new["recover_day"] == 12  # mild_days=(2,2) -> uniform(2,2)=2


def test_sick_severe(ut_sick, monkeypatch):
    monkeypatch.setitem(life.PROFILES, "ut_sick_severe", UT_SICK_SEVERE)
    st = _state(life.HEALTHY)
    new, event = life.daily_roll("张三", 30, 10, st, "ut_sick_severe", dice_seed=1)
    assert event == {"type": "sick", "severity": "severe"}
    assert new["health"] == life.SEVERE
    assert new["recover_day"] == 13  # severe_days=(3,3)


def test_mild_recover_on_due_day(ut_zero):
    # 轻症分支先掷「转重骰」(硬编码 0.15)，未中且到期 -> 痊愈
    seed = _dice_seed_where(lambda s: life._rng(s, "李四", 10).random() >= 0.15)
    st = _state(life.MILD, since_day=9, recover_day=10)
    new, event = life.daily_roll("李四", 30, 10, st, ut_zero, dice_seed=seed)
    assert event == {"type": "recover"}
    assert new["health"] == life.HEALTHY


def test_mild_worsen(ut_zero):
    seed = _dice_seed_where(lambda s: life._rng(s, "李四", 10).random() < 0.15)
    st = _state(life.MILD, since_day=9, recover_day=11)
    new, event = life.daily_roll("李四", 30, 10, st, ut_zero, dice_seed=seed)
    assert event == {"type": "worsen"}
    assert new["health"] == life.SEVERE
    assert new["recover_day"] == 12  # severe_days=(2,2)


def test_severe_recover_on_due_day(ut_zero):
    # 重症先判死（概率 0）再判到期痊愈
    st = _state(life.SEVERE, since_day=8, recover_day=10)
    new, event = life.daily_roll("王五", 30, 10, st, ut_zero, dice_seed=1)
    assert event == {"type": "recover"}
    assert new["health"] == life.HEALTHY


def test_severe_early_recover_by_dice(ut_zero):
    # 未到期时 35% 概率提前痊愈：第二次掷骰 < 0.35
    def _second_draw(s):
        rng = life._rng(s, "赵六", 10)
        rng.random()  # 第一次是死判（概率 0，必存活）
        return rng.random()

    seed = _dice_seed_where(lambda s: _second_draw(s) < 0.35)
    st = _state(life.SEVERE, since_day=8, recover_day=12)
    new, event = life.daily_roll("赵六", 30, 10, st, ut_zero, dice_seed=seed)
    assert event == {"type": "recover"}
    assert new["health"] == life.HEALTHY


def test_severe_continue(ut_zero):
    def _second_draw(s):
        rng = life._rng(s, "赵六", 10)
        rng.random()
        return rng.random()

    seed = _dice_seed_where(lambda s: _second_draw(s) >= 0.35)
    st = _state(life.SEVERE, since_day=8, recover_day=12)
    new, event = life.daily_roll("赵六", 30, 10, st, ut_zero, dice_seed=seed)
    assert event is None
    assert new["health"] == life.SEVERE


def test_severe_death(monkeypatch):
    monkeypatch.setitem(
        life.PROFILES, "ut_severe_die", dict(UT_ZERO, severe_die=[(999, 1.0)])
    )
    st = _state(life.SEVERE, since_day=8, recover_day=12)
    new, event = life.daily_roll("钱七", 70, 10, st, "ut_severe_die", dice_seed=1)
    assert event == {"type": "death", "cause": "重病"}
    assert new["health"] == life.DEAD


def test_healthy_natural_death(monkeypatch):
    monkeypatch.setitem(life.PROFILES, "ut_death", UT_DEATH)
    st = _state(life.HEALTHY)
    new, event = life.daily_roll("孙八", 70, 10, st, "ut_death", dice_seed=1)
    assert event == {"type": "death", "cause": "寿终"}
    assert new["health"] == life.DEAD


def test_dead_stays_dead(ut_zero):
    st = _state(life.DEAD)
    new, event = life.daily_roll("周九", 80, 10, st, ut_zero, dice_seed=1)
    assert event is None
    assert new["health"] == life.DEAD


def test_daily_roll_is_replayable(ut_sick):
    """同一 (种子, 角色, 日期) 重放：结果完全一致。"""
    def _run():
        st = _state(life.HEALTHY)
        return life.daily_roll("张三", 30, 10, st, ut_sick, dice_seed=42)

    new1, ev1 = _run()
    new2, ev2 = _run()
    assert ev1 == ev2
    assert new1 == new2


def test_daily_roll_order_independent(monkeypatch):
    """判定与调用顺序无关：先甲后乙 vs 先乙后甲，结果一一对应。"""
    monkeypatch.setitem(life.PROFILES, "ut_mixed", {
        "sick": [(999, 0.5)],
        "death": [(999, 0.2)],
        "severe_ratio": [(999, 0.3)],
        "severe_die": [(999, 0.5)],
        "mild_days": (1, 2),
        "severe_days": (2, 4),
    })

    def _run(order):
        out = []
        for name in order:
            st = _state(life.HEALTHY)
            out.append(life.daily_roll(name, 30, 5, st, "ut_mixed", dice_seed=9))
        return dict(zip(order, out))

    forward = _run(["甲", "乙"])
    backward = _run(["乙", "甲"])
    assert forward == backward


# ============ 婚配 ============

def test_marriage_roll_deterministic():
    assert life.marriage_roll(5, 42) == life.marriage_roll(5, 42)
    assert isinstance(life.marriage_roll(5, 42), bool)


def test_pick_pair_deterministic_and_valid():
    pairs = [("甲", "乙"), ("丙", "丁"), ("戊", "己")]
    p1 = life.pick_pair(pairs, 3, 7)
    p2 = life.pick_pair(pairs, 3, 7)
    assert p1 == p2 and p1 in pairs


def test_is_close_kin():
    # 直系：父亲/子女
    assert life.is_close_kin(["陈父"], [], "陈子", "陈父")
    assert life.is_close_kin([], ["陈父"], "陈父", "陈子")
    # 兄弟姐妹：共享任一亲本
    assert life.is_close_kin(["父", "母"], ["父", "母"], "兄", "妹")
    # 无关
    assert not life.is_close_kin([], [], "甲", "乙")
    assert not life.is_close_kin(["甲父"], ["乙母"], "甲", "乙")


# ============ 生育 ============

def test_can_conceive_boundaries():
    policy = {"adult_age": 18, "fertility_max_age": 45}
    assert not life.can_conceive(17.9, policy)
    assert life.can_conceive(18.0, policy)
    assert life.can_conceive(45.0, policy)
    assert not life.can_conceive(45.1, policy)


def test_conception_chance_bands():
    # 分段表按「年龄 < 上限」命中档位
    assert life.conception_chance(10) == pytest.approx(0.06)
    assert life.conception_chance(20) == pytest.approx(0.30)
    assert life.conception_chance(31) == pytest.approx(0.24)
    assert life.conception_chance(35) == pytest.approx(0.14)
    assert life.conception_chance(40) == pytest.approx(0.06)
    assert life.conception_chance(46) == pytest.approx(0.0)


def test_conception_roll_deterministic():
    assert life.conception_roll("夫", "妻", 5, 11) == life.conception_roll("夫", "妻", 5, 11)


def test_birth_chance_ok_window():
    # 1 模拟日 = 1/4 年 -> 生育窗口 0.25 模拟日
    assert life.birth_chance_ok([{"type": "birth", "day": 9.9}], 10, births_per_year=2)
    # 窗口内已有 2 个新生儿 -> 拒绝
    events = [{"type": "birth", "day": 9.9}, {"type": "birth", "day": 9.85}]
    assert not life.birth_chance_ok(events, 10, births_per_year=2)
    # 窗口外的 births 不计数
    events = [{"type": "birth", "day": 9.6}, {"type": "birth", "day": 9.7}]
    assert life.birth_chance_ok(events, 10, births_per_year=2)


def test_child_age():
    child = life.new_child_record(
        "李安", "男", ["父", "母"], birth_day=10.0, birth_frame=1,
        texture_from="父", home=["the Ville"],
    )
    assert life.child_age(child, 10.0) == 0.0
    assert life.child_age(child, 11.5) == pytest.approx(6.0)
    assert child["parents"] == ["父", "母"]


def test_new_life_state_defaults():
    st = life.new_life_state()
    assert st["health"] == life.HEALTHY
    assert st["spouse"] == "" and st["pregnant_by"] == ""
    assert st["births"] == 0 and st["parents"] == []


# ============ LLM 输出解析（叙事收敛层） ============

def test_parse_wedding_normalizes():
    text = '  "在鲜花与祝福中，\n 两人结为夫妻。"  '
    assert life.parse_wedding(text) == "在鲜花与祝福中， 两人结为夫妻。"


def test_parse_wedding_truncates():
    text = "一" * 300
    out = life.parse_wedding(text)
    assert len(out) == 221  # 220 字 + 省略号
    assert out.endswith("…")


def test_parse_wedding_empty():
    assert life.parse_wedding("") == ""
    assert life.parse_wedding(None) == ""


def test_parse_child_persona_canonical():
    text = (
        "先天：好奇、温和\n"
        "后天：在父母身边长大。\n"
        "生活习惯：作息规律。\n"
        "日常计划：白天上学。"
    )
    parsed = life.parse_child_persona(text)
    assert parsed == {
        "innate": "好奇、温和",
        "learned": "在父母身边长大。",
        "lifestyle": "作息规律。",
        "daily_plan": "白天上学。",
    }


def test_parse_child_persona_halfwidth_colon():
    text = "先天: 胆大心细\n后天: 自学成才\n生活习惯: 早睡早起\n日常计划: 上课读书"
    parsed = life.parse_child_persona(text)
    assert parsed["innate"] == "胆大心细"
    assert parsed["learned"] == "自学成才"


def test_parse_child_persona_fallback_innate():
    parsed = life.parse_child_persona("后天：在父母身边长大。", fallback_innate="沉稳、细心")
    assert parsed["innate"] == "沉稳、细心"
    assert parsed["learned"] == "在父母身边长大。"


def test_parse_child_persona_strips_bullets():
    text = "- 先天：好奇\n- 后天：读书\n- 生活习惯：早睡\n- 日常计划：上学"
    parsed = life.parse_child_persona(text)
    assert parsed["innate"] == "好奇"
    assert parsed["daily_plan"] == "上学"


# ============ 种子文件完整性（真实数据冒烟） ============

def test_seed_file_complete():
    seed = life.load_seed()
    life.parse_sim_time(seed["world"]["start"])          # 世界起点可解析
    assert seed["mortality"]["profile"] in life.PROFILES  # 档位存在
    assert isinstance(seed["mortality"]["dice_seed"], int)
    names = [a["name"] for a in seed["agents"]]
    assert len(names) == len(set(names)) >= 1             # 花名册唯一非空
    for key in (
        "cap", "births_per_year", "adult_age", "school_age",
        "marry_min_age", "fertility_max_age", "pregnancy_days",
    ):
        assert key in seed["population"]
    assert isinstance(seed["housing_pool"], list)
    assert life.seed_personas() == names
