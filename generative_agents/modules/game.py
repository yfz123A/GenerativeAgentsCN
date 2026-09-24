"""generative_agents.game"""

import os
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.utils import GenerativeAgentsMap, GenerativeAgentsKey
from modules import utils
from .maze import Maze
from .agent import Agent


class Game:
    """The Game"""

    def __init__(self, name, static_root, config, conversation, logger=None):
        self.name = name
        self.static_root = static_root
        self.record_iterval = config.get("record_iterval", 30)
        self.logger = logger or utils.IOLogger()
        self.maze = Maze(self.load_static(config["maze"]["path"]), self.logger)
        self.conversation = conversation
        self.agents = {}
        if "agent_base" in config:
            agent_base = config["agent_base"]
        else:
            agent_base = {}
        storage_root = os.path.join(f"results/checkpoints/{name}", "storage")
        if not os.path.isdir(storage_root):
            os.makedirs(storage_root)
        for name, agent in config["agents"].items():
            agent_config = utils.update_dict(
                copy.deepcopy(agent_base), self.load_static(agent["config_path"])
            )
            agent_config = utils.update_dict(agent_config, agent)

            agent_config["storage_root"] = os.path.join(storage_root, name)
            self.agents[name] = Agent(agent_config, self.maze, self.conversation, self.logger)

    def get_agent(self, name):
        return self.agents[name]

    def agent_info(self, name):
        """收集一个 Agent 本步的概要信息并写日志（与串行版完全一致）"""
        agent = self.get_agent(name)
        info = {
            "currently": agent.scratch.currently,
            "associate": agent.associate.abstract(),
            "concepts": {c.node_id: c.abstract() for c in agent.concepts},
            "chats": [
                {"name": "self" if n == agent.name else n, "chat": c}
                for n, c in agent.chats
            ],
            "action": agent.action.abstract(),
            "schedule": agent.schedule.abstract(),
            "address": agent.get_tile().get_address(as_list=False),
        }
        if (
            utils.get_timer().daily_duration() - agent.last_record
        ) > self.record_iterval:
            info["record"] = True
            agent.last_record = utils.get_timer().daily_duration()
        else:
            info["record"] = False
        if agent.llm_available():
            info["llm"] = agent._llm.get_summary()
        title = "{}.summary @ {}".format(
            name, utils.get_timer().get_date("%Y%m%d-%H:%M:%S")
        )
        self.logger.info("\n{}\n{}\n".format(utils.split_line(title), agent))
        return info

    def agent_think(self, name, status):
        agent = self.get_agent(name)
        plan = agent.think(status, self.agents)
        return {"plan": plan, "info": self.agent_info(name)}

    def agent_think_phases(self, statuses, parallel=1):
        """按阶段推进本步的所有 Agent（阶段化并行）。

        只把「纯角色私有状态 + LLM」的阶段放进线程池（日程重算 / 感知 / 反思），
        会写共享世界状态的阶段（移动、路径）与跨角色交互阶段（对话、等待）保持串行，
        所以并行的只是推理本身。

        与串行版的语义差别：所有 Agent 先各自移动、再各自感知/决策，即
        「同时行动」语义——互相看到的不是同一轮里已经做好的决策，而是上一轮的结果。
        parallel=1 时退化为完全串行（与 agent_think 逐条等价）。
        """
        names = list(statuses.keys())
        agents = self.agents
        results = {}

        def run_phase(fn):
            if parallel <= 1 or len(names) <= 1:
                for n in names:
                    fn(n)
                return
            with ThreadPoolExecutor(max_workers=min(parallel, len(names))) as pool:
                futures = {pool.submit(fn, n): n for n in names}
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc is not None:
                        raise RuntimeError(
                            "并行阶段失败（agent={}）".format(futures[fut])
                        ) from exc

        phase_cost = {}
        t0 = time.time()

        def mark(label):
            nonlocal t0
            now = time.time()
            phase_cost[label] = round(now - t0, 1)
            t0 = now

        for n in names:                                     # S0 移动（写迷宫，串行）
            agents[n].stage_move(statuses[n])
        mark("S0移动")
        run_phase(lambda n: agents[n].stage_schedule())      # S1 日程（LLM，并行）
        mark("S1日程")
        for n in names:                                     # S2 归位/醒睡（串行）
            agents[n].stage_settle()
        mark("S2归位")
        run_phase(lambda n: agents[n].stage_percept())       # S3 感知打分（LLM，并行）
        mark("S3感知")
        for n in names:                                     # S4 互动（对话/等待，跨角色，串行）
            agents[n].stage_plan(agents)
        mark("S4互动")
        run_phase(lambda n: agents[n].stage_decide_reflect())  # S5 行动决策+反思（LLM，并行）
        mark("S5决策/反思")
        for n in names:                                     # S6 路径与回放计划（串行）
            plan = agents[n].stage_finish(agents)
            results[n] = {"plan": plan, "info": self.agent_info(n)}
        mark("S6路径")

        self.logger.info(
            "阶段耗时（秒，并行度 {}）: {}".format(
                parallel,
                "  ".join("{}={}".format(k, v) for k, v in phase_cost.items()),
            )
        )
        return results

    def load_static(self, path):
        return utils.load_dict(os.path.join(self.static_root, path))

    def reset_game(self):
        for a_name, agent in self.agents.items():
            agent.reset()
            title = "{}.reset".format(a_name)
            self.logger.info("\n{}\n{}\n".format(utils.split_line(title), agent))


def create_game(name, static_root, config, conversation, logger=None):
    """Create the game"""

    utils.set_timer(**config.get("time", {}))
    GenerativeAgentsMap.set(GenerativeAgentsKey.GAME, Game(name, static_root, config, conversation, logger=logger))
    return GenerativeAgentsMap.get(GenerativeAgentsKey.GAME)


def get_game():
    """Get the gloabl game"""

    return GenerativeAgentsMap.get(GenerativeAgentsKey.GAME)
