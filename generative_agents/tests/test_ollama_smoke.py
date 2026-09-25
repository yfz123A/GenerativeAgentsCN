"""L4 真实冒烟测试（默认跳过）：需要本地 Ollama 在线。

运行方式：
    pytest -m ollama
"""

import json

import pytest

pytestmark = pytest.mark.ollama


def test_ollama_single_completion():
    from modules.model.llm_model import create_llm_model

    with open("data/config.json", "r", encoding="utf-8") as f:
        llm_config = json.load(f)["agent"]["think"]["llm"]

    model = create_llm_model(llm_config)
    out = model.completion("请只回复两个字：在线", retry=1, caller="smoke")
    assert isinstance(out, str)
    assert len(out.strip()) > 0
