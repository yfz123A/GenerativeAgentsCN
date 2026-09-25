"""pytest 全局夹具。

- 任何目录启动 pytest 都先 chdir 到 generative_agents/（产品代码以相对路径
  访问 data/、frontend/static/、results/）；
- 每个测试前后复位进程级单例（GenerativeAgentsMap）；
- fake_llm：按 func_hint/caller 键控的假 LLM + 假 embedding，全程离线；
- sim_name / static_agents_guard：测试产物不污染仓库。
"""

import os
import sys
import uuid
from pathlib import Path

import pytest

GENERATIVE_ROOT = Path(__file__).resolve().parents[1]
if str(GENERATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATIVE_ROOT))
os.chdir(GENERATIVE_ROOT)

from modules.utils.namespace import GenerativeAgentsMap  # noqa: E402


def _rmtree(path):
    """尽力而为的递归删除。

    本环境的 sitecustomize 把 os.remove/os.rmdir/shutil.rmtree 重定向到
    回收站机制，删除虽会执行，但其记账代码可能抛 SystemExit 等异常；
    且 /tmp 与工作区跨文件系统，rename 出工作区会 EXDEV。
    因此先改名成同级 .trash-*（一定成功，立即从真实路径消失），再尽力
    删除内容；任何清理失败都不应影响测试结果。
    """
    if not os.path.exists(path):
        return
    trashed = path
    try:
        trashed = "{}.trash-{}".format(path, uuid.uuid4().hex[:8])
        os.rename(path, trashed)
    except OSError:
        trashed = path
    try:
        for root, dirs, files in os.walk(trashed, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except BaseException:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except BaseException:
                    pass
        os.rmdir(trashed)
    except BaseException:
        pass


def _purge_stale_trash():
    """会话启动时清理上次测试遗留的 .trash-* 目录（尽力而为）。"""
    parent = GENERATIVE_ROOT / "results/checkpoints"
    if not os.path.isdir(parent):
        return
    for name in os.listdir(parent):
        if name.startswith(".trash-"):
            _rmtree(os.path.join(parent, name))


_purge_stale_trash()


@pytest.fixture(autouse=True)
def clean_global_state():
    """复位进程级单例，避免测试间相互污染。"""
    GenerativeAgentsMap.reset()
    yield
    GenerativeAgentsMap.reset()


@pytest.fixture(scope="session")
def start_module():
    """导入 start.py —— 它在模块层执行 argparse，需要先打桩 sys.argv。"""
    old_argv = sys.argv
    sys.argv = ["start.py"]
    try:
        import start
    finally:
        sys.argv = old_argv
    return start


@pytest.fixture
def sim_name():
    """唯一模拟局名；测试结束清理 Game 硬编码写入 results/ 的存档目录。"""
    name = "ut-" + uuid.uuid4().hex[:8]
    yield name
    _rmtree(GENERATIVE_ROOT / "results/checkpoints" / name)


@pytest.fixture
def static_agents_guard():
    """记录 agents 资产目录快照；测试结束后删除新增目录（生育/入学的角色资产）。"""
    agents_dir = GENERATIVE_ROOT / "frontend/static/assets/village/agents"
    before = {p.name for p in agents_dir.iterdir()}
    yield
    for p in agents_dir.iterdir():
        if p.name not in before:
            try:
                if p.is_dir():
                    _rmtree(p)
                else:
                    p.unlink()
            except BaseException:
                pass


@pytest.fixture
def fake_llm(monkeypatch):
    """离线替身：Agent LLM、生命链路 LLM、embedding 三处全部替换，零侵入产品代码。"""
    from fakes.llm import FakeLLM

    fake = FakeLLM()
    # 绑定方法挂到类属性后调用方会以 agent.completion(...) 形式调用，
    # 需要用普通函数桥接，把 Agent 实例显式转交给替身
    monkeypatch.setattr(
        "modules.agent.Agent.completion",
        lambda agent, func_hint, *args, **kwargs: fake.agent_completion(
            func_hint, agent, *args, **kwargs
        ),
    )
    monkeypatch.setattr(
        "modules.model.llm_model.LLMModel.completion", fake.llm_completion
    )
    monkeypatch.setattr(
        "modules.storage.index.OllamaEmbedding", fake.embedding_factory
    )
    return fake
