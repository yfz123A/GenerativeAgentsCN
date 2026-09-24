import os
import json
import argparse
from datetime import datetime

from modules.maze import Maze
from start import personas

file_markdown = "simulation.md"
file_movement = "movement.json"
life_events_file = "life_events.json"

frames_per_step = 60  # 每个step包含的帧数

maze_path = "frontend/static/assets/village/maze.json"
seed_path = "data/seed.json"


def load_seed():
    """读种子文件（角色性别 / 基准年龄 / 生命节奏）；缺失时返回空结构。"""
    try:
        with open(seed_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"agents": [], "world": {}}


# 将address转换为字符串
def get_location(address):
    # 仅为兼容原版
    # if address[0] == "<waiting>" or address[0] == "<persona>":
    #     return None

    # 不需要显示address第一级（"the Ville"）
    location = "，".join(address[1:])

    return location


class MovementBuilder:
    """回放帧构建器。

    compress.py 用它一次性处理全部存档，replay.py 用它做实时增量处理，
    两条链路共用同一份逻辑，以保证「实时流」与「事后压缩」的产出完全一致。

    增量用法：反复调用 add_step() 喂入新的存档文件即可。跨步状态
    （每个 Agent 的落点 _last_location、地图对象 _maze）会在实例内延续，
    这正是增量结果能与全量结果逐字节对齐的前提。
    """

    def __init__(self):
        self.stride = 1
        self.start_datetime = ""
        self.persona_init_pos = {}
        self.all_movement = {"description": {}, "conversation": {}}
        self.step = 0
        self._last_location = {}
        self._maze = None
        # 花名册：每个角色出现在哪些步（用于回放时决定何时生成/何时变成墓碑）
        self.agent_steps = {}
        # 运行时权威值（最后一步存档里的 life.spouse / homes）：
        # 种子夫妻没有婚礼事件，婚姻与住所要以运行时状态为准
        self._runtime = {}

    def _get_maze(self):
        if self._maze is None:
            with open(maze_path, "r", encoding="utf-8") as f:
                self._maze = Maze(json.load(f), None)
        return self._maze

    # 插入第0帧数据（Agent的初始状态）
    def _insert_frame0(self, agent_name, fallback=None):
        """insert frame0 for agent.

        fallback: 该角色没有静态设定文件时的兜底数据（旧存档里的角色目录可能已被
        改名或移除，此时退回用存档内记录的坐标，保证旧录像仍然可以重放）。
        """
        key = "0"
        if key not in self.all_movement.keys():
            self.all_movement[key] = dict()

        json_path = f"frontend/static/assets/village/agents/{agent_name}/agent.json"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            address = json_data["spatial"]["address"]["living_area"]
            location = get_location(address)
            coord = json_data["coord"]
            currently = json_data["currently"]
            scratch = json_data["scratch"]
        except (FileNotFoundError, KeyError):
            if not fallback or "coord" not in fallback:
                return
            coord = fallback["coord"]
            location = ""
            currently = ""
            scratch = {}

        self.persona_init_pos[agent_name] = coord
        self.all_movement[key][agent_name] = {
            "location": location,
            "movement": coord,
            "description": "正在睡觉",
        }
        self.all_movement["description"][agent_name] = {
            "currently": currently,
            "scratch": scratch,
        }

    # 处理单个存档文件，产出该步的 frames_per_step 帧
    def add_step(self, json_data, conversation=None):
        if conversation is None:
            conversation = {}

        if "stride" in json_data:
            self.stride = json_data["stride"]

        step = json_data["step"]
        agents = json_data["agents"]

        # 保存回放的起始时间
        if len(self.start_datetime) < 1:
            t = datetime.strptime(json_data["time"], "%Y%m%d-%H:%M")
            self.start_datetime = t.isoformat()

        # 遍历单个存档文件中的所有Agent
        for agent_name, agent_data in agents.items():
            # 插入第0帧
            if step == 1:
                self._insert_frame0(agent_name, agent_data)

            frame0_agent = self.all_movement.get("0", {}).get(agent_name, {})
            base_agent = self._last_location.get(agent_name, frame0_agent)
            source_coord = base_agent.get("movement", agent_data["coord"])
            target_coord = agent_data["coord"]
            location = get_location(agent_data["action"]["event"]["address"])
            if location is None:
                location = base_agent.get("location", "")
                path = [source_coord]
            else:
                path = self._get_maze().find_path(source_coord, target_coord)

            had_conversation = False
            step_conversation = ""
            persons_in_conversation = []
            step_time = json_data["time"]
            if step_time in conversation.keys():
                for chats in conversation[step_time]:
                    for persons, chat in chats.items():
                        persons_in_conversation.append(persons.split(" @ ")[0].split(" -> "))
                        step_conversation += f"\n地点：{persons.split(' @ ')[1]}\n\n"
                        for c in chat:
                            agent = c[0]
                            text = c[1]
                            step_conversation += f"{agent}：{text}\n"

            for i in range(frames_per_step):
                moving = len(path) > 1
                if len(path) > 0:
                    movement = list(path[0])
                    path = path[1:]
                    if agent_name not in self._last_location.keys():
                        self._last_location[agent_name] = dict()
                    self._last_location[agent_name]["movement"] = movement
                    self._last_location[agent_name]["location"] = location
                else:
                    movement = None

                if moving:
                    action = f"前往 {location}"
                elif movement is not None:
                    action = agent_data["action"]["event"]["describe"]
                    if len(action) < 1:
                        action = f'{agent_data["action"]["event"]["predicate"]}{agent_data["action"]["event"]["object"]}'

                    # 判断该存档文件中当前Agent是否有新的对话（用于设置图标）
                    for persons in persons_in_conversation:
                        if agent_name in persons:
                            had_conversation = True
                            break

                    # 针对睡觉和对话设置图标
                    if "睡觉" in action:
                        action = "😴 " + action
                    elif had_conversation:
                        action = "💬 " + action

                step_key = "%d" % ((step-1) * frames_per_step + 1 + i)
                if step_key not in self.all_movement.keys():
                    self.all_movement[step_key] = dict()

                if movement is not None:
                    self.all_movement[step_key][agent_name] = {
                        "location": location,
                        "movement": movement,
                        "action": action,
                    }
            self.all_movement["conversation"][step_time] = step_conversation

            # 侧边栏「近期状况 / 身体状况」取运行时状态（反复覆盖，最终即最后一步的值）。
            # 旧版只在第 0 帧写入静态人设，之后从不更新 —— 观众看到的永远是开局设定。
            detail = self.all_movement["description"].get(agent_name) or {}
            if agent_data.get("currently"):
                detail["currently"] = agent_data["currently"]
            life_state = agent_data.get("life") or {}
            if life_state.get("health"):
                detail["health"] = life_state["health"]
            self.all_movement["description"][agent_name] = detail

            # 记录运行时权威的配偶与住所（最后一步的值）：种子夫妻没有婚礼事件，
            # 搬家/再婚也只体现在存档里
            rt = self._runtime.setdefault(agent_name, {})
            if life_state.get("spouse"):
                rt["spouse"] = life_state["spouse"]
            homes = json_data.get("homes") or {}
            if agent_name in homes and homes[agent_name]:
                rt["home"] = list(homes[agent_name])

        # 花名册：记录每个角色首次/最后一次出现的步（生死决定何时出现、何时变墓碑）
        for agent_name in agents.keys():
            record = self.agent_steps.setdefault(agent_name, [step, step])
            record[0] = min(record[0], step)
            record[1] = max(record[1], step)

        self.step = step
        return step

    def add_life_events(self, events):
        """并入生命事件（死亡/出生），供 roster 使用。"""
        self.life_events = list(events or [])

    def roster(self):
        """生成花名册：每个角色的存在区间，以及死亡/出生信息。

        帧号规则与 all_movement 一致：第 N 步覆盖帧 [(N-1)*60+1, N*60]，第 0 帧为初始状态。
        另带静态属性（性别 / 基准年龄），供回放侧边栏计算实时年龄与展示。
        """
        seed = {a["name"]: a for a in load_seed().get("agents", [])}
        events = getattr(self, "life_events", [])
        roster = {}
        for agent_name, (first_step, last_step) in self.agent_steps.items():
            first_frame = 0 if first_step <= 1 else (first_step - 1) * frames_per_step + 1
            entry = {
                "born_frame": 0,           # 出现在侧边栏的帧（初始居民 = 0）
                "first_frame": first_frame,  # 出现在地图上的帧
                "last_frame": last_step * frames_per_step,
            }
            info = seed.get(agent_name)
            if info:
                entry["gender"] = info.get("gender", "")
                entry["base_age"] = float(info.get("age", 0))
                entry.setdefault("spouse", info.get("spouse") or "")   # 种子婚配（开局即夫妻）
                if info.get("home"):
                    entry.setdefault("home", list(info["home"]))       # 种子住所
            roster[agent_name] = entry

        for event in events:
            name = event.get("name")
            if event.get("type") == "death":
                entry = roster.setdefault(name, {})
                entry.setdefault("first_frame", 0)
                step = event.get("step")
                if step is not None:
                    # 死亡发生在该步开始前：最后存活帧 = 上一步的末帧
                    entry["last_frame"] = max(0, (step - 1) * frames_per_step)
                entry["died_of"] = event.get("cause", "重病")
                entry["age_at_death"] = event.get("age")
                if event.get("coord"):
                    entry["coord"] = event["coord"]
            elif event.get("type") == "birth":
                # 出生：从这一帧起出现在侧边栏；婴幼儿不在地图上走动（等入学才进场），
                # 所以 first_frame 先给一个「永不到达」的哨兵值，enroll 时再覆盖。
                entry = roster.setdefault(name, {})
                step = event.get("step", 1)
                entry["born_frame"] = (step - 1) * frames_per_step + 1
                entry["first_frame"] = 1 << 30
                entry.setdefault("last_frame", self.step * frames_per_step)
                if event.get("parents"):
                    entry["parents"] = event["parents"]
                if event.get("gender"):
                    entry["gender"] = event["gender"]
                if event.get("coord"):
                    entry["coord"] = event["coord"]
                entry["infant"] = True
            elif event.get("type") == "enroll":
                # 入学：从这一帧起成为完整角色，出现在地图上
                entry = roster.setdefault(name, {})
                step = event.get("step", 1)
                entry["first_frame"] = (step - 1) * frames_per_step + 1
                entry.pop("infant", None)
                entry.setdefault("last_frame", self.step * frames_per_step)
            elif event.get("type") == "marriage":
                # 婚配：双方互相登记配偶与新家（不改变存在区间，仅供展示）
                for who, other in ((name, event.get("spouse")), (event.get("spouse"), name)):
                    if not who or not other:
                        continue
                    who_entry = roster.setdefault(who, {})
                    who_entry["spouse"] = other
                    who_entry["married_frame"] = (event.get("step", 1) - 1) * frames_per_step + 1
                    if event.get("home"):
                        who_entry["home"] = event["home"]
        # 运行时最终状态覆盖（搬家/再婚以最后一步存档为准）
        for name, rt in self._runtime.items():
            entry = roster.setdefault(name, {})
            if rt.get("spouse"):
                entry["spouse"] = rt["spouse"]
            if rt.get("home"):
                entry["home"] = rt["home"]
        return roster

    def first_seen_coord(self, agent_name):
        """这个角色第一次出现在画面上的坐标（用于给中途入场的人定初始位置）。"""
        for frame_no in range(0, self.step * frames_per_step + 1):
            frame = self.all_movement.get(str(frame_no))
            if frame and agent_name in frame:
                return frame[agent_name]["movement"]
        return None

    def result(self):
        roster = self.roster()
        # 让花名册里的每个人都有初始位置：中途出生/入学的角色也要能建精灵
        # （前端先生成精灵、再由 apply_roster 按帧号决定是否可见）
        init_pos = dict(self.persona_init_pos)
        for agent_name in roster.keys():
            if agent_name in init_pos:
                continue
            coord = self.first_seen_coord(agent_name)
            if coord is None:
                entry = roster.get(agent_name) or {}
                coord = entry.get("coord")
            if coord is not None:
                init_pos[agent_name] = coord
        return {
            "start_datetime": self.start_datetime,  # 起始时间
            "stride": self.stride,  # 每个step对应的分钟数（必须与生成时的参数一致）
            "sec_per_step": self.stride,  # 回放时每一帧对应的秒数
            "years_per_sim_day": load_seed().get("world", {}).get("years_per_sim_day", 4),  # 生命节奏：1 模拟日 = 多少岁
            "persona_init_pos": init_pos,  # 每个Agent的初始位置（含中途入场者）
            "roster": roster,  # 花名册（出生/入学/死亡区间，回放据此生成、显隐与墓碑化）
            "all_movement": self.all_movement,  # 所有Agent在每个setp中的位置变化
        }


# 从所有存档文件中提取数据（用于回放）
def generate_movement(checkpoints_folder, compressed_folder, compressed_file):
    movement_file = os.path.join(compressed_folder, compressed_file)

    conversation_file = "conversation.json"
    conversation = {}
    if os.path.exists(os.path.join(checkpoints_folder, conversation_file)):
        with open(os.path.join(checkpoints_folder, conversation_file), "r", encoding="utf-8") as f:
            conversation = json.load(f)

    files = sorted(os.listdir(checkpoints_folder))
    json_files = list()
    for file_name in files:
        if file_name.endswith(".json") and file_name not in (conversation_file, life_events_file):
            json_files.append(os.path.join(checkpoints_folder, file_name))

    builder = MovementBuilder()
    # 生命事件（死亡/出生）：用于花名册与墓碑
    life_events_path = os.path.join(checkpoints_folder, life_events_file)
    if os.path.exists(life_events_path):
        with open(life_events_path, "r", encoding="utf-8") as f:
            builder.add_life_events(json.load(f))
    for file_name in json_files:
        # 依次读取所有存档文件
        with open(file_name, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        builder.add_step(json_data, conversation)

    result = builder.result()

    # 保存数据
    with open(movement_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))

    return result


# 生成Markdown文档
def generate_report(checkpoints_folder, compressed_folder, compressed_file):
    last_state = dict()

    conversation_file = "conversation.json"
    conversation = {}
    if os.path.exists(os.path.join(checkpoints_folder, conversation_file)):
        with open(os.path.join(checkpoints_folder, conversation_file), "r", encoding="utf-8") as f:
            conversation = json.load(f)

    def checkpoints_personas():
        """从存档第一步读花名册（角色可能已被改名或移除，不能依赖代码里的名单）"""
        files = sorted(
            f for f in os.listdir(checkpoints_folder)
            if f.endswith(".json") and f not in (conversation_file, life_events_file)
        )
        if len(files) < 1:
            return personas
        with open(os.path.join(checkpoints_folder, files[0]), "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("agents", {}).keys()) or personas

    def extract_description():
        markdown_content = "# 基础人设\n\n"
        # 花名册从存档来（旧存档的角色可能已被改名或移除，不能依赖代码里的名单）
        for agent_name in checkpoints_personas():
            json_path = f"frontend/static/assets/village/agents/{agent_name}/agent.json"
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
            except (FileNotFoundError, KeyError):
                continue
            markdown_content += f"## {agent_name}\n\n"
            markdown_content += f"年龄：{json_data['scratch']['age']}岁  \n"
            markdown_content += f"先天：{json_data['scratch']['innate']}  \n"
            markdown_content += f"后天：{json_data['scratch']['learned']}  \n"
            markdown_content += f"生活习惯：{json_data['scratch']['lifestyle']}  \n"
            markdown_content += f"当前状态：{json_data['currently']}\n\n"
        return markdown_content

    def extract_action(json_data):
        markdown_content = ""
        agents = json_data["agents"]
        for agent_name, agent_data in agents.items():
            if agent_name not in last_state.keys():
                last_state[agent_name] = {"currently": "", "location": "", "action": ""}

            location = "，".join(agent_data["action"]["event"]["address"])
            action = agent_data["action"]["event"]["describe"]

            if location == last_state[agent_name]["location"] and action == last_state[agent_name]["action"]:
                continue

            last_state[agent_name]["location"] = location
            last_state[agent_name]["action"] = action

            if len(markdown_content) < 1:
                markdown_content = f"# {json_data['time']}\n\n"
                markdown_content += "## 活动记录：\n\n"

            markdown_content += f"### {agent_name}\n"

            if len(action) < 1:
                action = "睡觉"

            markdown_content += f"位置：{location}  \n"
            markdown_content += f"活动：{action}  \n"

            markdown_content += f"\n"

        if json_data['time'] not in conversation.keys():
            return markdown_content

        markdown_content += "## 对话记录：\n\n"
        for chats in conversation[json_data['time']]:
            for agents, chat in chats.items():
                markdown_content += f"### {agents}\n\n"
                for item in chat:
                    markdown_content += f"`{item[0]}`\n> {item[1]}\n\n"
        return markdown_content

    all_markdown_content = extract_description()
    files = sorted(os.listdir(checkpoints_folder))
    for file_name in files:
        if (not file_name.endswith(".json")) or (file_name in (conversation_file, life_events_file)):
            continue

        file_path = os.path.join(checkpoints_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            content = extract_action(json_data)
            all_markdown_content += content + "\n\n"
    with open(f"{compressed_folder}/{compressed_file}", "w", encoding="utf-8") as compressed_file:
        compressed_file.write(all_markdown_content)


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="the name of the simulation")
args = parser.parse_args()


if __name__ == "__main__":
    name = args.name
    if len(name) < 1:
        name = input("Please enter a simulation name: ")

    while not os.path.exists(f"results/checkpoints/{name}"):
        name = input(f"'{name}' doesn't exists, please re-enter the simulation name: ")

    checkpoints_folder = f"results/checkpoints/{name}"
    compressed_folder = f"results/compressed/{name}"
    os.makedirs(compressed_folder, exist_ok=True)

    generate_report(checkpoints_folder, compressed_folder, file_markdown)
    generate_movement(checkpoints_folder, compressed_folder, file_movement)

    print("Compression completed.")
