"""generative_agents.modules.life

种子与生命时钟。

设计要点（与 2026-09-24 的设计评审一致）：
- 花名册从数据（data/seed.json）来，不写死在代码里；
- 抽象年龄是「模拟时间的纯函数」：age = 基准年龄 + 经过的模拟日 × years_per_sim_day
  （1 模拟日 = 4 岁）。不落盘、不需要存档字段，任何时刻可由 world.start 推出，
  与行为时钟（stride 分钟/步）完全解耦；
- 判定层参数（人口政策、死亡档位、骰子种子）全部收在种子里，改基调只改数据。
"""

import json
import random
from datetime import datetime

seed_file = "data/seed.json"

_seed_cache = None


def load_seed(path=seed_file):
    """加载种子（进程内缓存一份）。"""
    global _seed_cache
    if _seed_cache is None:
        with open(path, "r", encoding="utf-8") as f:
            _seed_cache = json.load(f)
    return _seed_cache


def seed_personas(path=seed_file):
    """初始花名册（按种子顺序）。"""
    return [a["name"] for a in load_seed(path)["agents"]]


def parse_sim_time(s):
    """'20250213-09:30' -> datetime"""
    return datetime.strptime(s, "%Y%m%d-%H:%M")


def world_start(path=seed_file):
    """模拟世界的起始时刻（datetime）。"""
    return parse_sim_time(load_seed(path)["world"]["start"])


def years_per_sim_day(path=seed_file):
    return float(load_seed(path)["world"].get("years_per_sim_day", 4))


def population_policy(path=seed_file):
    """人口政策参数包。"""
    return load_seed(path)["population"]


def mortality_profile(path=seed_file):
    """死亡判定参数包（档位与骰子种子）。"""
    return load_seed(path)["mortality"]


def sim_days_elapsed(cur_dt, start_dt=None, path=seed_file):
    """从世界起点到 cur_dt 经过的模拟日数（浮点）。"""
    if start_dt is None:
        start_dt = world_start(path)
    return (cur_dt - start_dt).total_seconds() / 86400.0


def abstract_age(base_age, cur_dt, start_dt=None, path=seed_file):
    """角色在 cur_dt 时刻的抽象年龄。

    base_age 是种子里的基准年龄（world.start 时刻的年龄）。
    """
    return base_age + sim_days_elapsed(cur_dt, start_dt, path) * years_per_sim_day(path)


# ============ 生死判定引擎（骰子层） ============
#
# 设计原则（2026-09-24 评审 Q3c/Q9b）：
#   - 判定与叙事分离：这里只掷骰子产出「事件」，措辞交给 LLM；
#   - 全部随机数由 (dice_seed, 角色名, 模拟日) 派生 —— 同一存档重放，
#     生死线完全一致，方便调试与复现；
#   - 参数表集中在下方，改基调只改这里；预期寿命用 monte_carlo 校准。

# 每模拟日的事件概率表：按抽象年龄分段
#   sick     : 当日发病概率（发病后按 severe_ratio 决定轻重）
#   death    : 当日自然死亡概率（寿终，与疾病无关，老年才显著）
#   severe_die: 重症者当日死亡概率
# 数值由 monte_carlo 校准（四项约束同时满足）：
#   预期寿命        gentle≈80 / standard≈70 / harsh≈55 岁
#   重病死因占比    50% / 66% / 78%（其余为寿终）
#   人均发病        1.30 / 1.45 / 1.52 次每生（避免「半个镇天天生病」）
#   全城事件密度    25 人小镇约 1.6 / 2.1 / 2.8 起发病每模拟日，其中重症约 0.45 / 0.8 / 1.4 起
#                   （一个模拟日 = 144 步 ≈ 2 小时真实计算；实际密度随年龄结构浮动）
PROFILES = {
    "gentle": {
        "sick": [(18, 0.029), (40, 0.0435), (60, 0.0725), (75, 0.116), (999, 0.174)],
        "death": [(59, 0.0), (69, 0.0145), (79, 0.0508), (89, 0.1305), (999, 0.261)],
        "severe_ratio": [(18, 0.20), (60, 0.30), (999, 0.42)],
        "severe_die": [(59, 0.35), (74, 0.45), (999, 0.55)],
        "mild_days": (1, 2),          # 轻症自愈所需的模拟日数区间
        "severe_days": (2, 4),        # 重症痊愈所需模拟日数区间
    },
    "standard": {
        "sick": [(18, 0.0417), (40, 0.0625), (60, 0.1042), (75, 0.1668), (999, 0.2502)],
        "death": [(59, 0.0), (69, 0.0208), (79, 0.0695), (89, 0.1668), (999, 0.3336)],
        "severe_ratio": [(18, 0.25), (60, 0.35), (999, 0.50)],
        "severe_die": [(59, 0.50), (74, 0.60), (999, 0.70)],
        "mild_days": (1, 2),
        "severe_days": (2, 4),
    },
    "harsh": {
        "sick": [(18, 0.0738), (40, 0.1066), (60, 0.1722), (75, 0.2624), (999, 0.3936)],
        "death": [(49, 0.0), (59, 0.0246), (69, 0.0656), (79, 0.1558), (89, 0.3116), (999, 0.45)],
        "severe_ratio": [(18, 0.30), (60, 0.45), (999, 0.60)],
        "severe_die": [(59, 0.60), (74, 0.70), (999, 0.80)],
        "mild_days": (1, 2),
        "severe_days": (2, 4),
    },
}

# 健康状态
HEALTHY, MILD, SEVERE, DEAD = "healthy", "mild", "severe", "dead"

# 每模拟日的受孕概率（按女方年龄）：婚后每天掷一次骰子
CONCEPTION = [(18, 0.06), (25, 0.30), (32, 0.24), (38, 0.14), (45, 0.06), (999, 0.0)]


def new_life_state():
    """一个角色的完整生命状态（随存档持久化）。"""
    return {
        "health": HEALTHY,
        "since_day": 0.0,     # 当前状态开始于第几个模拟日（浮点）
        "recover_day": 0.0,   # 计划痊愈的模拟日
        "sick_count": 0,      # 累计患病次数（叙事用）
        "spouse": "",         # 配偶姓名（空 = 未婚）
        "pregnant_by": "",    # 怀孕时记下孩子父亲（空 = 未孕）
        "due_day": 0.0,       # 预产日
        "births": 0,          # 已生育子女数
        "parents": [],        # 父母姓名（新生儿才有）
    }


def _band(table, age):
    """按年龄查分段表。"""
    for limit, value in table:
        if age < limit:
            return value
    return table[-1][1]


def _rng(dice_seed, agent_name, day_index):
    """确定性骰子：同一存档、同一角色、同一天 -> 同一结果。"""
    return random.Random("{}:{}:{}".format(dice_seed, agent_name, int(day_index)))


def daily_roll(agent_name, age, day_index, state, profile="standard", dice_seed=0):
    """推进一天，返回 (新状态, 事件)。

    事件形如 {"type": "sick"/"recover"/"worsen"/"death", ...}，没有变化则返回 None。
    判定完全由 (dice_seed, agent_name, day_index) 决定，与调用顺序无关。
    """
    conf = PROFILES.get(profile, PROFILES["standard"])
    rng = _rng(dice_seed, agent_name, day_index)
    health = state["health"]

    if health == DEAD:
        return state, None

    # 1) 重症：先判死，再判痊愈，否则持续
    if health == SEVERE:
        if rng.random() < _band(conf["severe_die"], age):
            state["health"] = DEAD
            state["since_day"] = day_index
            return state, {"type": "death", "cause": "重病"}
        if day_index >= state["recover_day"] or rng.random() < 0.35:
            state["health"] = HEALTHY
            state["since_day"] = day_index
            return state, {"type": "recover"}
        return state, None

    # 2) 轻症：到期自愈，小概率转重
    if health == MILD:
        if rng.random() < 0.15:
            state["health"] = SEVERE
            state["since_day"] = day_index
            state["recover_day"] = day_index + rng.uniform(*conf["severe_days"])
            return state, {"type": "worsen"}
        if day_index >= state["recover_day"]:
            state["health"] = HEALTHY
            state["since_day"] = day_index
            return state, {"type": "recover"}
        return state, None

    # 3) 健康：先判自然死亡（寿终），再判发病
    if rng.random() < _band(conf["death"], age):
        state["health"] = DEAD
        state["since_day"] = day_index
        return state, {"type": "death", "cause": "寿终"}

    if rng.random() < _band(conf["sick"], age):
        severe = rng.random() < _band(conf["severe_ratio"], age)
        state["health"] = SEVERE if severe else MILD
        state["since_day"] = day_index
        state["sick_count"] += 1
        days = conf["severe_days"] if severe else conf["mild_days"]
        state["recover_day"] = day_index + rng.uniform(*days)
        return state, {"type": "sick", "severity": state["health"]}

    return state, None


# ---------- 生育 ----------
def can_conceive(age, policy, path=seed_file):
    """是否处于育龄窗口（18~45 岁）。"""
    return policy["adult_age"] <= age <= policy["fertility_max_age"]


def conception_roll(husband, wife, day_index, dice_seed):
    """确定性受孕骰子：同一存档、同一对夫妻、同一天结果一致。"""
    rng = _rng(dice_seed, "conceive:{}:{}".format(husband, wife), day_index)
    return rng.random()


def conception_chance(age):
    return _band(CONCEPTION, age)


def birth_chance_ok(events, day_index, births_per_year=3):
    """每模拟年（= 1/years_per_sim_day 个模拟日）最多 N 个新生儿。"""
    window = 1.0 / years_per_sim_day()
    recent = [e for e in events
              if e.get("type") == "birth" and day_index - e.get("day", 0) < window]
    return len(recent) < births_per_year


def new_child_record(name, gender, parents, birth_day, birth_frame, texture_from, home):
    """幼儿登记：0~6 岁不进模拟循环，只作为「被照顾对象」存在。"""
    return {
        "name": name,
        "gender": gender,
        "parents": list(parents),
        "birth_day": birth_day,       # 抽象日（浮点）
        "birth_frame": birth_frame,   # 出生帧（花名册用）
        "texture_from": texture_from,  # 贴图继承自哪位父母
        "home": list(home),
    }


def child_age(child, day_index):
    """幼儿当前的抽象年龄。"""
    return (day_index - child["birth_day"]) * years_per_sim_day()


# ============ 婚配判定（骰子层） ============
#
# 设计（评审 Q4c「分层」）：骰子负责「筛候选 + 决定成婚」，LLM 只写一场婚礼叙事
# （一次性、有预算）。与生死判定同样是确定性的：同一存档重放，婚配线一致。
MARRIAGE_CHANCE = 0.35   # 每天在适龄单身者中促成一对婚礼的概率


def marriage_roll(day_index, dice_seed):
    """确定性骰子：今天是否举办婚礼。"""
    return _rng(dice_seed, "marriage", day_index).random() < MARRIAGE_CHANCE


def pick_pair(pairs, day_index, dice_seed):
    """从合法配对中确定性地选一对。"""
    rng = _rng(dice_seed, "couple", day_index)
    return pairs[rng.randrange(len(pairs))]


def is_close_kin(parents_a, parents_b, name_a, name_b):
    """禁止近亲：直系（父母/子女）与兄弟姐妹（共享任一亲本）。"""
    if name_a in parents_b or name_b in parents_a:
        return True
    return bool(set(parents_a) & set(parents_b))


def wedding_prompt(groom, bride, groom_scratch, bride_scratch):
    """婚礼叙事：一次 LLM 调用，产出小镇公告式的短叙事。"""
    return (
        "小镇上，{groom} 与 {bride} 决定结为夫妻。\n"
        "{groom}：{g_in}；{g_learned}\n"
        "{bride}：{b_in}；{b_learned}\n"
        "请以小镇见闻的口吻写一段简短的婚礼叙事（2~3 句）：可以写婚礼的场面、"
        "到场亲友的祝福，或两人当时的心情。\n"
        "只输出叙事本身，不要标题、不要引号、不要换行。"
    ).format(
        groom=groom, bride=bride,
        g_in=groom_scratch.get("innate", ""), g_learned=groom_scratch.get("learned", ""),
        b_in=bride_scratch.get("innate", ""), b_learned=bride_scratch.get("learned", ""),
    )


def parse_wedding(text, limit=220):
    """把 LLM 输出收敛成一句干净的叙事；空输出返回空串（由调用方兜底）。"""
    if not text:
        return ""
    text = " ".join(text.split())
    text = text.strip().strip("\"'“”「」")
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def child_name_prompt(father, mother, gender, existing_names=None):
    """让 LLM 起名：姓氏随父，符合中国习惯。"""
    used = "、".join(existing_names or [])
    return (
        "小镇里有一对新婚夫妇即将迎来他们的孩子。\n"
        "父亲叫{father}，母亲叫{mother}。\n"
        "请按中国习惯给孩子起一个名字：姓氏随父亲，名字用 1~2 个常用汉字，"
        "风格朴实自然，避免生僻字与叠字，不要与下列已有居民重名：{used}。\n"
        "孩子性别：{gender}。\n"
        "只输出名字本身，不要任何解释、标点或引号。"
    ).format(father=father, mother=mother, gender=gender, used=used)


def child_persona_prompt(name, gender, father, mother, father_scratch, mother_scratch):
    """入学转正时让 LLM 依据父母撰写这个孩子的人设。"""
    return (
        "请为一个即将入学的小镇居民撰写人物设定。\n"
        "姓名：{name}\n性别：{gender}\n父亲：{father}（{f_in}；{f_learned}）\n"
        "母亲：{mother}（{m_in}；{m_learned}）\n"
        "这个孩子从小在父母身边长大，性格与习惯会同时带有一点父母的特征，"
        "但也已经长成独立的个体。\n"
        "请严格按下面四行输出，每行以标签开头，不要多余内容：\n"
        "先天：<用 2~4 个词概括的性格与天赋，顿号分隔>\n"
        "后天：<一两句话描述成长经历与所学会的技能>\n"
        "生活习惯：<一句话描述作息与生活习惯>\n"
        "日常计划：<一句话描述入学后白天常去的地方与活动>"
    ).format(
        name=name, gender=gender, father=father, mother=mother,
        f_in=father_scratch.get("innate", ""), f_learned=father_scratch.get("learned", ""),
        m_in=mother_scratch.get("innate", ""), m_learned=mother_scratch.get("learned", ""),
    )


def parse_child_persona(text, fallback_innate="好奇、温和"):
    """解析 LLM 返回的四行人设；解析失败时退化为父母的混搭。"""
    result = {
        "innate": "", "learned": "", "lifestyle": "", "daily_plan": "",
    }
    labels = {
        "先天": "innate", "后天": "learned",
        "生活习惯": "lifestyle", "日常计划": "daily_plan",
    }
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*").strip()
        for label, key in labels.items():
            prefix = label + "："
            if line.startswith(prefix):
                result[key] = line[len(prefix):].strip()
            elif line.startswith(label + ":"):
                result[key] = line[len(label) + 1:].strip()
    if not result["innate"]:
        result["innate"] = fallback_innate
    return result


def monte_carlo(profile="standard", n=20000, max_age=120.0, dice_seed=0):
    """校准用：从 0 岁开始逐日掷骰到死，返回期望寿命与死因分布。"""
    conf = PROFILES.get(profile, PROFILES["standard"])
    total, causes = 0.0, {"重病": 0, "寿终": 0}
    for i in range(n):
        age, day, state = 0.0, 0.0, new_life_state()
        # 每天推进 4 岁（与 years_per_sim_day 一致）
        while age < max_age:
            state, event = daily_roll("mc-{}".format(i), age, day, state,
                                      profile, dice_seed)
            if event and event["type"] == "death":
                causes[event["cause"]] = causes.get(event["cause"], 0) + 1
                break
            age += 4.0
            day += 1.0
        total += age
    return total / n, causes
