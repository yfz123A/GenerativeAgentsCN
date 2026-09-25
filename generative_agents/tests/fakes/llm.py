"""测试替身：fake LLM 与 fake embedding（键控应答）。

键控策略：
- Agent 的全部 LLM 调用都经过 Agent.completion(func_hint, ...)，这里按
  func_hint 返回「最终形态」的应答（即真实链路里 callback 之后、写进业务
  逻辑的值），让正常决策路径真实走通；
- 生命链路（婚礼叙事/起名/人设）由 start.py 直接调 LLMModel.completion
  (caller=...)，这里按 caller 键控返回纯文本；
- embedding 走 modules.storage.index.OllamaEmbedding，替换为 llama_index
  的 MockEmbedding（确定性定长向量，构造与查询全程离线）。
"""

# ---------- 生命链路应答（按 caller 键控） ----------
LIFE_RESPONSES = {
    "life_wedding": "在亲友的见证下，两人在小镇教堂结为夫妻，大家都为他们感到高兴。",
    "life_name": "李安",
    "life_persona": (
        "先天：好奇、温和\n"
        "后天：在父母身边长大，学会了读书写字。\n"
        "生活习惯：作息规律。\n"
        "日常计划：白天去学校上课，放学后回家。"
    ),
}

# ---------- Agent 应答（按 func_hint 键控） ----------

# 与真实 prompt_schedule_daily 的 failsafe 同构：键覆盖 wake_up..23 点，
# 值的不同取值数 >= schedule.diversity(5)，且白天不含「睡」字。
DAILY_SCHEDULE = {
    "7:00": "起床并完成早晨的例行工作",
    "8:00": "在学院上课",
    "12:00": "吃午饭",
    "13:00": "在图书馆读书",
    "17:00": "在公园散步",
    "18:00": "回家吃晚饭",
    "19:00": "在家看电视",
    "22:00": "准备睡觉",
    "23:00": "睡觉",
}


def _pick_leaf(spatial, address):
    """从空间树确定性地选一个「有子节点」的分支，保证最终地址在迷宫里真实存在。"""
    options = sorted(spatial.get_leaves(address))
    for name in options:
        if spatial.get_leaves(list(address) + [name]):
            return name
    return options[0] if options else ""


def _agent_response(func_hint, agent, *args, **kwargs):
    if func_hint == "wake_up":
        return 7
    if func_hint == "schedule_init":
        return [
            "早上7点起床并吃早餐",
            "上午去学院上课",
            "中午吃午饭",
            "下午在图书馆读书",
            "傍晚回家吃晚饭",
            "晚上11点睡觉",
        ]
    if func_hint == "schedule_daily":
        wake_up = args[0]
        return {
            "{h}:00".format(h=h): DAILY_SCHEDULE.get("{h}:00".format(h=h), "休息")
            for h in range(wake_up, 24)
        }
    if func_hint == "schedule_decompose":
        plan, _schedule = args
        return [(plan["describe"], plan["duration"])]
    if func_hint == "schedule_revise":
        _action, schedule = args
        plan, _ = schedule.current_plan()
        return plan.get("decompose") or [
            {
                "idx": 0,
                "describe": plan["describe"],
                "start": plan["start"],
                "duration": plan["duration"],
            }
        ]
    if func_hint == "retrieve_plan":
        return ["{name} 有既定的日程安排".format(name=agent.name)]
    if func_hint == "retrieve_thought":
        return "{name} 想起了近期的安排".format(name=agent.name)
    if func_hint == "retrieve_currently":
        return agent.scratch.currently
    if func_hint == "reflect_focus":
        return ["{name} 今天要做什么？".format(name=agent.name)]
    if func_hint == "reflect_insights":
        nodes = args[0]
        return [["{name} 有了新的感悟".format(name=agent.name), [nodes[0].node_id]]]
    if func_hint == "reflect_chat_planing":
        return "这次对话让{name}的计划更清晰了".format(name=agent.name)
    if func_hint == "reflect_chat_memory":
        return "{name} 记住了这次对话".format(name=agent.name)
    if func_hint == "determine_sector":
        return _pick_leaf(kwargs["spatial"], kwargs["address"])
    if func_hint == "determine_arena":
        return _pick_leaf(kwargs["spatial"], kwargs["address"])
    if func_hint == "determine_object":
        options = sorted(kwargs["spatial"].get_leaves(kwargs["address"]))
        return options[0] if options else ""
    if func_hint == "describe_object":
        return "安静"
    if func_hint == "decide_chat":
        return False
    if func_hint == "decide_chat_terminate":
        return True
    if func_hint == "decide_wait":
        return False
    if func_hint == "generate_chat":
        return "你好呀，今天过得怎么样？"
    if func_hint == "generate_chat_check_repeat":
        return False
    if func_hint == "summarize_relation":
        return "{name} 认识 {other}".format(name=agent.name, other=args[1])
    if func_hint == "summarize_chats":
        return "聊了聊近况"
    if func_hint == "poignancy_chat":
        return 3
    if func_hint == "poignancy_event":
        return 2
    raise AssertionError("FakeLLM: 未实现 func_hint 的应答: " + func_hint)


class FakeLLM:
    """三处替身的载体，并记录全部调用便于测试断言。"""

    def __init__(self):
        self.agent_calls = []   # [(agent_name, func_hint)]
        self.llm_calls = []     # [{"caller", "prompt_head"}]
        self.embedding_builds = 0

    def agent_completion(self, func_hint, agent, *args, **kwargs):
        # 与真实 Agent.completion 相同的契约：func_hint 必须有对应 prompt 模板
        assert hasattr(agent.scratch, "prompt_" + func_hint), (
            "FakeLLM: scratch 上找不到 prompt_{}".format(func_hint)
        )
        self.agent_calls.append((agent.name, func_hint))
        return _agent_response(func_hint, agent, *args, **kwargs)

    def llm_completion(self, prompt, retry=10, callback=None, failsafe=None,
                       return_type=None, caller="llm_normal", **kwargs):
        self.llm_calls.append({"caller": caller, "prompt_head": prompt[:60]})
        text = LIFE_RESPONSES.get(caller)
        if text is None:
            raise AssertionError("FakeLLM: 未实现 caller 的应答: " + caller)
        return callback(text) if callback else text

    def embedding_factory(self, *args, **kwargs):
        self.embedding_builds += 1
        try:
            from llama_index.core.embeddings import MockEmbedding
        except ImportError:  # 兼容不同版本的导出位置
            from llama_index.core.embeddings.mock_embed import MockEmbedding
        return MockEmbedding(embed_dim=8)
