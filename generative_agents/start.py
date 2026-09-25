import os
import sys
import copy
import json
import time
import shutil
import argparse
import datetime

from dotenv import load_dotenv, find_dotenv

from modules.game import create_game, get_game
from modules import utils
from modules import life
from modules import memory
from modules.life import seed_personas

# 花名册从种子来（data/seed.json）；性别/婚配/生死等生命参数见种子文件
personas = seed_personas()

# 生命事件文件名（与 compress.py 保持一致）；它不是存档，恢复时应跳过
life_events_name = "life_events.json"


class SimulateServer:
    def __init__(self, name, static_root, checkpoints_folder, config, start_step=0, verbose="info", log_file="", parallel=1, min_free_gb=5.0):
        self.name = name
        self.static_root = static_root
        self.checkpoints_folder = checkpoints_folder
        self.parallel = parallel
        # 挂机相关：磁盘看门狗阈值 + 结束原因（watchdog 据此判断要不要重新拉起）
        self.min_free_gb = float(min_free_gb)
        self.exit_reason = None
        self.last_step = start_step

        # 历史存档数据（用于断点恢复）
        self.config = config

        os.makedirs(checkpoints_folder, exist_ok=True)

        # 载入历史对话数据（用于断点恢复）
        self.conversation_log = f"{checkpoints_folder}/conversation.json"
        if os.path.exists(self.conversation_log):
            with open(self.conversation_log, "r", encoding="utf-8") as f:
                conversation = json.load(f)
        else:
            conversation = {}

        if len(log_file) > 0:
            self.logger = utils.create_file_logger(f"{checkpoints_folder}/{log_file}", verbose)
        else:
            self.logger = utils.create_io_logger(verbose)

        # 创建游戏
        game = create_game(name, static_root, config, conversation, logger=self.logger)
        game.reset_game()

        self.game = get_game()
        self.tile_size = self.game.maze.tile_size
        self.agent_status = {}
        if "agent_base" in config:
            agent_base = config["agent_base"]
        else:
            agent_base = {}
        for agent_name, agent in config["agents"].items():
            agent_config = copy.deepcopy(agent_base)
            agent_config.update(self.load_static(agent["config_path"]))
            self.agent_status[agent_name] = {
                "coord": agent_config["coord"],
                "path": [],
            }
        self.think_interval = max(
            a.think_config["interval"] for a in self.game.agents.values()
        )
        self.start_step = start_step

        # ---------- 生命循环（生死判定） ----------
        seed = life.load_seed()
        self.mortality_profile = life.mortality_profile()["profile"]
        self.dice_seed = life.mortality_profile()["dice_seed"]
        self.agent_base_age = {a["name"]: float(a["age"]) for a in seed["agents"]}
        # 生命原点：新建时记为种子起点（或本次 --start），断点续跑沿用它，
        # 这样抽象年龄始终连续，不会因续跑而回到 0。
        if "life_origin" not in config:
            config["life_origin"] = seed["world"]["start"]
        self.life_origin = life.parse_sim_time(config["life_origin"])
        # 生命状态随存档恢复（断点续跑不会重置病情）
        self.life_state = {}
        for agent_name in self.agent_status.keys():
            saved = config["agents"].get(agent_name, {}).get("life")
            self.life_state[agent_name] = saved or life.new_life_state()
        # 初始婚姻关系来自种子（只在新开局时写入，之后随存档走）
        seed_spouse = {a["name"]: a.get("spouse") for a in seed["agents"]}
        for agent_name, state in self.life_state.items():
            if not state.get("spouse") and seed_spouse.get(agent_name):
                state["spouse"] = seed_spouse[agent_name]
        self.bind_health_to_prompt()
        # 已经过去了几个模拟日（恢复时不能重掷已判定的日子）
        self.life_day = int(life.sim_days_elapsed(utils.get_timer().get_date(), self.life_origin))
        self.life_events_file = os.path.join(checkpoints_folder, life_events_name)
        self.life_events = []
        if os.path.exists(self.life_events_file):
            with open(self.life_events_file, "r", encoding="utf-8") as f:
                self.life_events = json.load(f)

        # ---------- 生育与成长 ----------
        self.population = life.population_policy()
        # 幼儿登记表（0~6 岁：只在侧边栏，不进模拟循环），随存档持久化
        self.children = config.get("children", [])
        # 性别表（从静态设定读一次；新生儿登记时写入）
        self.agent_gender = {}
        for agent_name in self.agent_status.keys():
            self.agent_gender[agent_name] = self.read_gender(agent_name)
        for child in self.children:
            self.agent_gender[child["name"]] = child.get("gender", "")

        # ---------- 婚配与住房 ----------
        # 运行时住址表：搬家只改它（agent.json 保持初始设定不被改写），随存档持久化
        self.housing_pool = [list(x) for x in config.get(
            "housing_pool", life.load_seed().get("housing_pool", [])
        )]
        self.homes = {}
        saved_homes = config.get("homes") or {}
        for agent_name in self.agent_status.keys():
            self.homes[agent_name] = list(
                saved_homes.get(agent_name) or self.read_home(agent_name)
            )
        for agent_name, home in saved_homes.items():   # 断点续跑：把已搬迁的家应用回角色
            if home:
                self.apply_home(agent_name, home)
        self._life_llm = None

    def apply_home(self, agent_name, home):
        """把住址写进运行中的角色（spatial.address + 派生的睡觉地址）。"""
        agent = self.game.agents.get(agent_name)
        if not agent:
            return
        agent.spatial.address["living_area"] = list(home)
        agent.spatial.address["睡觉"] = list(home) + ["床"]

    def read_gender(self, agent_name):
        path = f"frontend/static/assets/village/agents/{agent_name}/agent.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("gender", "")
        except (FileNotFoundError, json.JSONDecodeError):
            return ""

    def read_home(self, agent_name):
        """角色的初始住址（静态设定里的 living_area）。"""
        path = f"frontend/static/assets/village/agents/{agent_name}/agent.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)["spatial"]["address"]["living_area"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return []

    def read_scratch(self, agent_name):
        """角色的静态设定（先天/后天等），用于婚礼与人设生成。"""
        path = f"frontend/static/assets/village/agents/{agent_name}/agent.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("scratch", {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def life_llm(self):
        """起名 / 撰写人设专用的小 LLM 句柄（与角色同源配置）。"""
        if self._life_llm is None:
            from modules.model.llm_model import create_llm_model
            self._life_llm = create_llm_model(self.config["agent_base"]["think"]["llm"])
        return self._life_llm

    # ---------- 生命循环：把健康状态暴露给 prompt ----------
    def bind_health_to_prompt(self):
        """把「生病/康复」写进角色的当前状态，让 LLM 自己演出卧床/虚弱的行为。"""
        for name, state in self.life_state.items():
            agent = self.game.agents.get(name)
            if not agent:
                continue
            note = self.health_note(state)
            agent.scratch.currently = self.strip_health(agent.scratch.currently) + note

    @staticmethod
    def health_note(state):
        if state["health"] == life.SEVERE:
            return "（卧床：身患重病，虚弱）"
        if state["health"] == life.MILD:
            return "（不适：有些小病，精神不佳）"
        return ""

    @staticmethod
    def strip_health(text):
        for note in ("（卧床：身患重病，虚弱）", "（不适：有些小病，精神不佳）"):
            text = text.replace(note, "")
        return text

    def life_tick(self, timer, step=None):
        """按模拟日推进生死判定；返回本步内发生的生命事件。"""
        now = timer.get_date()
        day = int(life.sim_days_elapsed(now, self.life_origin))
        if day <= self.life_day:
            return []
        self.life_day = day
        events = []
        for name in list(self.agent_status.keys()):
            state = self.life_state.setdefault(name, life.new_life_state())
            base_age = self.agent_base_age.get(name, 30.0)
            age = life.abstract_age(base_age, now, self.life_origin)
            state, event = life.daily_roll(
                name, age, day, state, self.mortality_profile, self.dice_seed
            )
            if not event:
                continue
            event = dict(
                event,
                name=name,
                day=day,
                step=(step or 1),
                age=round(age, 1),
                time=now.strftime("%Y%m%d-%H:%M"),
            )
            events.append(event)
            if event["type"] == "death":
                self.kill_agent(name, event)
            else:
                self.logger.info(
                    "{} 的健康事件：{}（第 {} 模拟日，年龄 {}）".format(
                        name, event["type"], day, event["age"]
                    )
                )
                if event["type"] in ("sick", "worsen", "recover"):
                    agent = self.game.agents.get(name)
                    if agent:
                        agent.scratch.currently = (
                            self.strip_health(agent.scratch.currently)
                            + self.health_note(state)
                        )
        # 婚配链路：配对 → 婚礼叙事 → 搬家（婚后即打开生育窗口）
        events.extend(self.marriage_tick(timer, day, step))
        # 生育链路：受孕 → 分娩 → 入学（同一天内推进）
        events.extend(self.family_tick(timer, day, step))

        if len(events) > 0:
            self.life_events.extend(events)
            self.save_life_events()
        return events

    def kill_agent(self, name, event):
        """角色死亡：清出世界（迷宫事件、游戏循环、存档），留下墓碑事件。"""
        agent = self.game.agents.get(name)
        coord = list(agent.coord) if agent and agent.coord else None
        if agent:
            tile = agent.get_tile()
            tile.remove_events(subject=name)
            if tile.has_address("game_object"):
                addr = tile.get_address("game_object")
                self.game.maze.update_obj(
                    agent.coord, memory.Event(addr[-1], address=addr)
                )
        event.update({"coord": coord, "cause": event.get("cause", "重病")})
        self.game.agents.pop(name, None)
        self.agent_status.pop(name, None)
        self.config["agents"].pop(name, None)
        self.logger.info(
            "{} 去世了（享年 {} 岁，死因 {}，位置 {}）".format(
                name, event["age"], event["cause"], coord
            )
        )

    def save_life_events(self):
        atomic_write_json(self.life_events_file, self.life_events)

    # ---------- 婚配 ----------
    def marriage_tick(self, timer, day, step):
        """一天推进一次：适龄单身者配对 → LLM 婚礼叙事 → 搬家（婚后受孕窗口自动打开）。"""
        now = timer.get_date()
        if not life.marriage_roll(day, self.dice_seed):
            return []

        # 1) 适龄单身者：无在世配偶、已达婚龄；按姓名排序保证骰子可复现
        singles = {"男": [], "女": []}
        for name in sorted(self.agent_status.keys()):
            state = self.life_state.get(name)
            if state is None or state["health"] == life.DEAD:
                continue
            spouse = state.get("spouse") or ""
            if spouse and spouse in self.agent_status:
                continue                     # 有在世配偶
            age = life.abstract_age(self.agent_base_age.get(name, 30.0), now, self.life_origin)
            if age < self.population["marry_min_age"]:
                continue
            gender = self.agent_gender.get(name)
            if gender in singles:
                singles[gender].append(name)
        if not singles["男"] or not singles["女"]:
            return []

        # 2) 合法配对（排除近亲），骰子选一对
        pairs = []
        for groom in singles["男"]:
            gp = self.life_state[groom].get("parents") or []
            for bride in singles["女"]:
                bp = self.life_state[bride].get("parents") or []
                if life.is_close_kin(gp, bp, groom, bride):
                    continue
                pairs.append((groom, bride))
        if not pairs:
            return []
        groom, bride = life.pick_pair(pairs, day, self.dice_seed)

        # 3) 婚礼叙事（一次 LLM 调用，失败则用兜底文案）
        narrative = self.generate_wedding(groom, bride)

        # 4) 结为夫妻 —— family_tick 的受孕判定会随「有在世配偶」自动生效
        self.life_state[groom]["spouse"] = bride
        self.life_state[bride]["spouse"] = groom

        # 5) 搬家：优先住房池，池空则搬入男方住处
        home = self.assign_home(groom, bride)

        # 6) 婚礼见闻写进对话记录 —— 直播台的「对话记录」栏会显示它
        key = now.strftime("%Y%m%d-%H:%M")
        self.game.conversation.setdefault(key, []).append({
            "{} 与 {} @ 小镇教堂".format(groom, bride): [["小镇公告", narrative]]
        })

        event = {
            "type": "marriage", "name": groom, "spouse": bride,
            "day": day, "step": step, "age": None,
            "home": home, "narrative": narrative,
            "time": now.strftime("%Y%m%d-%H:%M"),
        }
        self.logger.info(
            "{} 与 {} 结为夫妻（新家：{}）—— {}".format(
                groom, bride, "，".join(home or []), narrative
            )
        )
        return [event]

    def generate_wedding(self, groom, bride):
        """LLM 婚礼叙事；失败或空输出时退回一句朴素文案。"""
        try:
            prompt_text = life.wedding_prompt(
                groom, bride, self.read_scratch(groom), self.read_scratch(bride)
            )
            text = self.life_llm().completion(prompt_text, retry=3, caller="life_wedding")
            narrative = life.parse_wedding(text)
            if narrative:
                return narrative
        except Exception as e:
            self.logger.info("婚礼叙事生成失败，改用兜底文案：{}".format(e))
        return "{}与{}在小镇教堂举行了简单的婚礼，亲友们都来道贺。".format(groom, bride)

    def assign_home(self, groom, bride):
        """新婚住房：优先分配住房池的空房，池空则搬入男方（或女方）现有住所。"""
        home = None
        if self.housing_pool:
            home = list(self.housing_pool.pop(0))
        if not home:
            home = list(self.homes.get(groom) or self.homes.get(bride) or [])
        if not home:
            return None
        for who in (groom, bride):
            self.homes[who] = list(home)
            self.apply_home(who, home)
        return home

    # ---------- 生育与成长 ----------
    def family_tick(self, timer, day, step):
        """一个模拟日推进一次：受孕 → 分娩 → 入学。"""
        now = timer.get_date()
        events = []
        alive = list(self.agent_status.keys())

        # 1) 受孕：已婚、女方在世且在育龄窗口、未孕、未达人口上限
        for name in alive:
            state = self.life_state.get(name)
            if state is None or state["health"] == life.DEAD:
                continue
            if self.agent_gender.get(name) != "女":
                continue
            if state.get("pregnant_by") or not state.get("spouse"):
                continue
            spouse = state["spouse"]
            if spouse not in self.agent_status:
                continue  # 配偶已故
            age = life.abstract_age(self.agent_base_age.get(name, 30.0), now, self.life_origin)
            if not life.can_conceive(age, self.population):
                continue
            if (len(self.agent_status) + len(self.children)) >= self.population["cap"]:
                continue
            if not life.birth_chance_ok(self.life_events, day, self.population["births_per_year"]):
                continue
            if life.conception_roll(spouse, name, day, self.dice_seed) < life.conception_chance(age):
                state["pregnant_by"] = spouse
                state["due_day"] = day + self.population["pregnancy_days"]
                event = {
                    "type": "conceive", "name": name, "spouse": spouse,
                    "day": day, "step": step, "age": round(age, 1),
                    "time": now.strftime("%Y%m%d-%H:%M"),
                }
                events.append(event)
                self.logger.info("{} 有了身孕（{} 的孩子）".format(name, spouse))

        # 2) 分娩
        for name in alive:
            state = self.life_state.get(name)
            if not state or not state.get("pregnant_by"):
                continue
            if day < state.get("due_day", 0):
                continue
            event = self.give_birth(name, state, day, step, now)
            if event:
                events.append(event)

        # 3) 入学：幼儿长到 school_age 转为完整角色
        for child in list(self.children):
            age = life.child_age(child, day)
            if age < self.population["school_age"]:
                continue
            self.enroll_child(child, day, step, now)
            events.append({
                "type": "enroll", "name": child["name"], "day": day, "step": step,
                "age": round(age, 1), "time": now.strftime("%Y%m%d-%H:%M"),
            })

        return events

    def give_birth(self, mother_name, state, day, step, now):
        """分娩：起名（姓氏随父）→ 登记幼儿 → 继承父母贴图。"""
        father_name = state["pregnant_by"]
        gender = "女" if life._rng(self.dice_seed, "gender:" + mother_name, day).random() < 0.5 else "男"
        base_age = life.abstract_age(self.agent_base_age.get(mother_name, 30.0), now, self.life_origin)

        name = self.generate_child_name(father_name, mother_name, gender, base_age)
        if not name or name in self.agent_status or any(c["name"] == name for c in self.children):
            name = self.fallback_child_name(father_name)

        texture_from = mother_name if self.agent_gender.get(mother_name) == gender else father_name
        if texture_from not in self.agent_status:
            texture_from = mother_name
        home = self.child_home(mother_name, father_name)

        record = life.new_child_record(
            name, gender, [father_name, mother_name], day,
            (step - 1) * 60 + 1, texture_from, home,
        )
        self.children.append(record)
        self.agent_gender[name] = gender
        self.write_child_assets(record)

        state["pregnant_by"] = ""
        state["due_day"] = 0.0
        state["births"] = state.get("births", 0) + 1
        father_state = self.life_state.get(father_name)
        if father_state is not None:
            father_state["births"] = father_state.get("births", 0) + 1

        self.logger.info(
            "{} 生下了{}：{}（父亲 {}）".format(mother_name, gender, name, father_name)
        )
        home_coord = self.game.agents[mother_name].coord if mother_name in self.game.agents else None
        return {
            "type": "birth", "name": name, "gender": gender,
            "parents": [father_name, mother_name],
            "day": day, "step": step, "frame": record["birth_frame"],
            "coord": list(home_coord) if home_coord else None,
            "age": 0.0, "time": now.strftime("%Y%m%d-%H:%M"),
        }

    def generate_child_name(self, father_name, mother_name, gender, base_age):
        """LLM 起名：姓氏随父；失败则退回默认名。"""
        existing = list(self.agent_status.keys()) + [c["name"] for c in self.children]
        try:
            prompt_text = life.child_name_prompt(father_name, mother_name, gender, existing)
            text = self.life_llm().completion(prompt_text, retry=3, caller="life_name")
            if text:
                name = text.strip().splitlines()[0].strip().strip("。.，,、\"'“”")
                if 2 <= len(name) <= 4:
                    return name
        except Exception as e:
            self.logger.info("起名失败，使用备用名：{}".format(e))
        return ""

    def fallback_child_name(self, father_name):
        """LLM 不可用时的备用名：姓氏随父 + 常用字。"""
        surname = father_name[0] if father_name else "小"
        pool = ["安", "宁", "禾", "苗", "星", "舟", "然", "一", "乐", "和", "平", "嘉"]
        i = 0
        while True:
            name = "{}{}".format(surname, pool[i % len(pool)])
            if name not in self.agent_status and all(c["name"] != name for c in self.children):
                return name
            i += 1

    def child_home(self, mother_name, father_name):
        """孩子住在母亲（或父亲）当前的住所（搬家后以运行时住址表为准）。"""
        homes = getattr(self, "homes", None) or {}
        for who in (mother_name, father_name):
            home = homes.get(who) or self.read_home(who)   # 兜底：退回静态设定
            if home:
                return list(home)
        return ["the Ville"]

    def write_child_assets(self, child):
        """给新生儿准备一张立绘与贴图（继承自父/母）。"""
        dst = f"frontend/static/assets/village/agents/{child['name']}"
        src = f"frontend/static/assets/village/agents/{child['texture_from']}"
        if os.path.isdir(dst):
            return
        os.makedirs(dst, exist_ok=True)
        for fname in ("portrait.png", "texture.png"):
            if os.path.exists(os.path.join(src, fname)):
                shutil.copyfile(os.path.join(src, fname), os.path.join(dst, fname))

    def enroll_child(self, child, day, step, now):
        """入学转正：撰写人设 → 建档 → 加入模拟循环。"""
        name = child["name"]
        father, mother = (child["parents"] + ["", ""])[:2]
        scratch = self.generate_child_persona(child, father, mother)

        home = child["home"]
        # 优先站在母亲（或父亲）身边——孩子本就住在家里，这样也不会落到壁橱之类的格子上
        parent = next((p for p in (mother, father) if p in self.game.agents), None)
        if parent:
            coord = list(self.game.agents[parent].coord)
        else:
            tiles = sorted(self.game.maze.get_address_tiles(home))  # 返回的是 set，排序后取整
            coord = list(tiles[0]) if tiles else [0, 0]

        asset_dir = f"frontend/static/assets/village/agents/{name}"
        if not os.path.isdir(asset_dir):
            os.makedirs(asset_dir, exist_ok=True)
            for fname in ("portrait.png", "texture.png"):
                src = f"frontend/static/assets/village/agents/{child['texture_from']}/{fname}"
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(asset_dir, fname))
        asset_path = os.path.join(asset_dir, "agent.json")
        payload = {
            "name": name,
            "portrait": "assets/village/agents/{}/portrait.png".format(name),
            "gender": child["gender"],
            "coord": coord,
            "currently": "{}刚满六岁，今天是入学的第一天。".format(name),
            "scratch": {
                "age": int(self.population["school_age"]),
                "innate": scratch["innate"],
                "learned": scratch["learned"],
                "lifestyle": scratch["lifestyle"],
                "daily_plan": scratch["daily_plan"],
            },
            "spatial": {
                "address": {"living_area": home},
                "tree": self.child_spatial_tree(father, mother, home),
            },
        }
        with open(asset_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, ensure_ascii=False))

        # 加入游戏
        from modules.agent import Agent
        agent_base = self.config.get("agent_base", {})
        agent_config = utils.update_dict(copy.deepcopy(agent_base), payload)
        agent_config["storage_root"] = os.path.join(
            f"results/checkpoints/{self.name}", "storage", name
        )
        agent = Agent(agent_config, self.game.maze, self.game.conversation, self.logger)
        agent.reset()
        self.game.agents[name] = agent

        self.config["agents"][name] = {"config_path": os.path.join("assets", "village", "agents", name, "agent.json")}
        self.agent_status[name] = {"coord": coord, "path": []}
        self.agent_base_age[name] = float(self.population["school_age"])
        state = life.new_life_state()
        state["parents"] = [father, mother]
        self.life_state[name] = state
        self.children.remove(child)

        self.logger.info("{} 入学了，成为小镇的正式居民".format(name))

    def generate_child_persona(self, child, father, mother):
        """LLM 依父母撰写人设；失败则用父母的混搭兜底。"""
        def scratch_of(who):
            path = f"frontend/static/assets/village/agents/{who}/agent.json"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)["scratch"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                return {}

        f_scratch, m_scratch = scratch_of(father), scratch_of(mother)
        try:
            prompt_text = life.child_persona_prompt(
                child["name"], child["gender"], father, mother, f_scratch, m_scratch
            )
            text = self.life_llm().completion(prompt_text, retry=3, caller="life_persona")
            parsed = life.parse_child_persona(text, fallback_innate=f_scratch.get("innate", "好奇、温和"))
            if parsed.get("learned"):
                return parsed
        except Exception as e:
            self.logger.info("撰写人设失败，使用兜底：{}".format(e))
        merged = [f_scratch.get("innate", ""), m_scratch.get("innate", "")]
        return {
            "innate": "、".join([x for x in merged if x]) or "好奇、温和",
            "learned": "在{}和{}身边长大。".format(father, mother),
            "lifestyle": m_scratch.get("lifestyle", "作息规律。"),
            "daily_plan": "白天去学校上课，放学后回家。",
        }

    def child_spatial_tree(self, father, mother, home):
        """孩子的空间认知：继承父母的（同一座小镇，房间地址一致）。"""
        for who in (mother, father):
            path = f"frontend/static/assets/village/agents/{who}/agent.json"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)["spatial"]["tree"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                continue
        return {}

    def disk_free_gb(self):
        """存档盘剩余空间（GB）。"""
        probe = self.checkpoints_folder
        while probe and not os.path.exists(probe):
            probe = os.path.dirname(probe)
        try:
            return shutil.disk_usage(probe or ".").free / (1024 ** 3)
        except OSError:
            return float("inf")

    def simulate(self, step, stride=0, forever=False):
        """推进模拟。

        forever=True 时不设步数上限，直到全灭或磁盘看门狗触发才停；
        每一步结束都会原子写盘，所以随时可以被 kill / 崩溃后从最近存档续跑。
        """
        timer = utils.get_timer()
        limit = None if forever else self.start_step + step
        i = self.start_step
        while limit is None or i < limit:
            if forever:
                title = "Simulate Step[{}, time: {}]".format(i + 1, timer.get_date())
            else:
                title = "Simulate Step[{}/{}, time: {}]".format(i+1, limit, timer.get_date())
            self.logger.info("\n" + utils.split_line(title, "="))

            # 生命循环：跨模拟日时判定发病/痊愈/死亡（全灭则自动结束）
            self.life_tick(timer, step=i + 1)
            if len(self.agent_status) < 1:
                self.exit_reason = "extinct"
                self.logger.info("\n" + utils.split_line("小镇已无居民，模拟结束", "="))
                break

            step_start = time.time()
            if self.parallel > 1:
                # 阶段化并行：LLM 密集且只碰角色私有状态的阶段走线程池
                self.logger.info("本步并行度: {}".format(self.parallel))
                results = self.game.agent_think_phases(self.agent_status, self.parallel)
            else:
                results = {
                    name: self.game.agent_think(name, status)
                    for name, status in self.agent_status.items()
                }
            for name, status in self.agent_status.items():
                plan = results[name]["plan"]
                agent = self.game.get_agent(name)
                if name not in self.config["agents"]:
                    self.config["agents"][name] = {}
                self.config["agents"][name].update(agent.to_dict())
                if plan.get("path"):
                    status["coord"], status["path"] = plan["path"][-1], []
                self.config["agents"][name].update(
                    # {"coord": status["coord"], "path": plan["path"]}
                    {"coord": status["coord"]}
                )
                # 生命状态随存档持久化（断点续跑时不会重置病情）
                if name in self.life_state:
                    self.config["agents"][name]["life"] = self.life_state[name]

            sim_time = timer.get_date("%Y%m%d-%H:%M")
            self.config.update(
                {
                    "time": sim_time,
                    "step": i + 1,
                    # 幼儿登记表随存档持久化（0~6 岁不进模拟循环，只作被照顾对象）
                    "children": self.children,
                    # 婚配与住房：住址表与住房池（搬过家的人续跑后仍在各自的新家）
                    "homes": self.homes,
                    "housing_pool": self.housing_pool,
                }
            )
            # 保存Agent活动数据（原子写：崩溃也不会留下半个存档）
            atomic_write_json(
                os.path.join(self.checkpoints_folder, "simulate-{}.json".format(sim_time.replace(":", ""))),
                self.config,
            )
            # 保存对话数据
            atomic_write_json(
                os.path.join(self.checkpoints_folder, "conversation.json"),
                self.game.conversation,
            )

            self.last_step = i + 1
            self.logger.info(
                "本步耗时 {:.1f} 秒（并行度 {}）".format(time.time() - step_start, self.parallel)
            )

            if stride > 0:
                timer.forward(stride)

            # 磁盘看门狗：空间不足时优雅停止（当前步的存档已完整落盘）
            free_gb = self.disk_free_gb()
            if free_gb < self.min_free_gb:
                self.exit_reason = "disk"
                self.logger.info(
                    "磁盘剩余 {:.1f}GB 低于阈值 {:.1f}GB，停止模拟（已存档至第 {} 步）".format(
                        free_gb, self.min_free_gb, i + 1
                    )
                )
                break

            i += 1

        if self.exit_reason is None:
            self.exit_reason = "finished"
        self.logger.info(
            "\n"
            + utils.split_line(
                "模拟结束（{}）：共推进到第 {} 步".format(self.exit_reason, self.last_step), "="
            )
        )

    def load_static(self, path):
        return utils.load_dict(os.path.join(self.static_root, path))


def missing_agents(config, static_root="frontend/static"):
    """续跑前自检：存档里的角色是否都还有静态设定。

    角色改名（或静态目录被清理）后，旧存档会指向不存在的 agent.json，
    不检查的话会以一个难懂的 JSONDecodeError 崩掉；这里提前给出人话。

    返回找不到静态设定的角色名列表。
    """
    missing = []
    for agent_name, agent in (config.get("agents") or {}).items():
        rel = (agent or {}).get("config_path") or os.path.join(
            "assets", "village", "agents", agent_name.replace(" ", "_"), "agent.json"
        )
        if not os.path.exists(os.path.join(static_root, rel)):
            missing.append(agent_name)
    return missing


def atomic_write_json(path, data, indent=2):
    """原子写存档：先写临时文件再 replace，避免崩溃留下半个 JSON。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=indent, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# 从存档数据中载入配置，用于断点恢复
def get_config_from_log(checkpoints_folder):
    files = sorted(os.listdir(checkpoints_folder))

    json_files = list()
    for file_name in files:
        if file_name.endswith(".json") and file_name not in ("conversation.json", life_events_name):
            json_files.append(os.path.join(checkpoints_folder, file_name))

    if len(json_files) < 1:
        return None

    # 从最新往前找第一个「完整可用」的存档：崩溃可能留下半个文件、字段缺失或
    # 时间格式异常，挂机自动续跑时应当回退到上一条可用存档，而不是原地再崩一次。
    assets_root = os.path.join("assets", "village")
    for path in reversed(json_files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not isinstance(config, dict) or "agents" not in config or "time" not in config:
            continue
        try:
            start_time = datetime.datetime.strptime(config["time"], "%Y%m%d-%H:%M")
        except (ValueError, TypeError):
            continue
        start_time += datetime.timedelta(minutes=config.get("stride", 10))
        config["time"] = {"start": start_time.strftime("%Y%m%d-%H:%M")}
        for a in config["agents"]:
            config["agents"][a]["config_path"] = os.path.join(assets_root, "agents", a.replace(" ", "_"), "agent.json")
        return config
    return None


# 为新游戏创建配置
def get_config(start_time="20240213-09:30", stride=15, agents=None):
    with open("data/config.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
        agent_config = json_data["agent"]

    assets_root = os.path.join("assets", "village")
    config = {
        "stride": stride,
        "time": {"start": start_time},
        "maze": {"path": os.path.join(assets_root, "maze.json")},
        "agent_base": agent_config,
        "agents": {},
    }
    for a in agents:
        config["agents"][a] = {
            "config_path": os.path.join(
                assets_root, "agents", a.replace(" ", "_"), "agent.json"
            ),
        }
    return config


load_dotenv(find_dotenv())

parser = argparse.ArgumentParser(description="console for village")
parser.add_argument("--name", type=str, default="", help="The simulation name")
parser.add_argument("--start", type=str, default="20240213-09:30", help="The starting time of the simulated ville")
parser.add_argument("--resume", action="store_true", help="Resume running the simulation")
parser.add_argument("--step", type=int, default=10, help="The simulate step")
parser.add_argument("--stride", type=int, default=10, help="The step stride in minute")
parser.add_argument("--verbose", type=str, default="debug", help="The verbose level")
parser.add_argument("--log", type=str, default="", help="Name of the log file")
parser.add_argument(
    "--parallel",
    type=int,
    default=1,
    help="阶段化并行的线程数（1=完全串行，建议 4~8；需 Ollama 侧 OLLAMA_NUM_PARALLEL 同步调大）",
)
parser.add_argument(
    "--forever",
    action="store_true",
    help="无限推进（挂机）：不设步数上限，直到全灭或磁盘不足；配合 run_forever.py 可崩溃自动续跑",
)
parser.add_argument(
    "--min-free-gb",
    type=float,
    default=5.0,
    help="磁盘看门狗阈值（GB）：存档盘剩余低于该值时优雅停止，默认 5",
)
args = parser.parse_args()


if __name__ == "__main__":
    checkpoints_path = "results/checkpoints"

    name = args.name
    if len(name) < 1:
        name = input("Please enter a simulation name (e.g. sim-test): ")

    checkpoints_folder = f"{checkpoints_path}/{name}"
    start_time = args.start

    # 续跑判定：有可用存档就接着跑；没有（目录不存在、或里面只有半个文件的存档 ——
    # 挂机时第一步就崩溃正是这个状态）则降级为新局，避免监督器把「无处可续」当成正常结束。
    sim_config = None
    if args.resume:
        if os.path.isdir(checkpoints_folder):
            sim_config = get_config_from_log(checkpoints_folder)
        if sim_config is None:
            if args.forever:
                print("没有可用存档，按新局开始（--forever）。")
            else:
                while not os.path.exists(checkpoints_folder):
                    name = input(f"'{name}' doesn't exists, please re-enter the simulation name: ")
                    checkpoints_folder = f"{checkpoints_path}/{name}"
                sim_config = get_config_from_log(checkpoints_folder)
                if sim_config is None:
                    print("No checkpoint file found to resume running.")
                    exit(0)

    if sim_config is None:
        # 新局：交互式运行时保留重名保护；挂机（--forever）下直接复用目录，不阻塞
        if not args.forever:
            while os.path.exists(checkpoints_folder):
                name = input(f"The name '{name}' already exists, please enter a new name: ")
                checkpoints_folder = f"{checkpoints_path}/{name}"
        os.makedirs(checkpoints_folder, exist_ok=True)
        sim_config = get_config(start_time, args.stride, personas)
        start_step = 0
    else:
        start_step = sim_config["step"]
        missing = missing_agents(sim_config, static_root="frontend/static")
        if missing:
            print("无法续跑：存档里的这些角色在当前静态设定里找不到 ——")
            print("  {}".format("、".join(missing[:10]) + ("…" if len(missing) > 10 else "")))
            print("常见原因：角色改名后，旧存档与新的 agent.json 不再对应。")
            print("处理办法：换个新名字开新局；这份存档仍可用 compress.py 回放。")
            sys.exit(2)

    static_root = "frontend/static"

    server = SimulateServer(
        name,
        static_root,
        checkpoints_folder,
        sim_config,
        start_step,
        args.verbose,
        args.log,
        parallel=max(1, args.parallel),
        min_free_gb=args.min_free_gb,
    )
    server.simulate(args.step, args.stride, forever=args.forever)

    # 结束标记：给挂机监督器（run_forever.py）与人看的收尾行
    print("SIMULATION_END reason={} last_step={} name={}".format(
        server.exit_reason, server.last_step, name
    ))
