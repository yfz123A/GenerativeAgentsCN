"""L3 单元测试：OllamaLLMModel 的容错逻辑（mock requests.post，全程离线）。

评审决策：只测 Ollama 分支（现网在用）；OpenAI 分支是 magentic 薄封装，跳过。
"""

import pytest
from pydantic import BaseModel

from modules.model import llm_model as lm


class Echo(BaseModel):
    res: str


def make_model():
    return lm.OllamaLLMModel(
        {"api_key": "", "base_url": "http://127.0.0.1:11434/v1", "model": "ut-model"}
    )


class _Response:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _content_body(content):
    return {"choices": [{"message": {"content": content}}]}


def patch_post(monkeypatch, outcomes, capture=None):
    """替换 requests.post；outcomes 是异常/响应体的队列，逐次消耗。"""
    queue = list(outcomes)

    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        if capture is not None:
            capture.update({"url": url, "body": json})
        out = queue.pop(0) if queue else queue
        if isinstance(out, Exception):
            raise out
        return _Response(out)

    monkeypatch.setattr(lm.requests, "post", fake_post)
    return capture


def patch_sleep(monkeypatch):
    monkeypatch.setattr(lm.time, "sleep", lambda *_: None)


# ============ 基础链路 ============

def test_completion_plain_text(monkeypatch):
    patch_post(monkeypatch, [_content_body("你好")])
    model = make_model()
    assert model.completion("提示", caller="ut") == "你好"


def test_empty_choices_yields_none_unless_failsafe(monkeypatch):
    # _completion 返回 ""，completion 里 "" or failsafe -> None / failsafe
    patch_post(monkeypatch, [{}])
    model = make_model()
    assert model.completion("提示", caller="ut") is None

    patch_post(monkeypatch, [{}])
    model2 = make_model()
    assert model2.completion("提示", failsafe="兜底", caller="ut") == "兜底"


def test_request_payload_fields(monkeypatch):
    capture = patch_post(monkeypatch, [_content_body("ok")], capture={})
    model = make_model()
    model.completion("提示", caller="ut")
    body = capture["body"]
    assert capture["url"].endswith("/chat/completions")
    assert body["model"] == "ut-model"
    assert body["messages"] == [{"role": "user", "content": "提示"}]
    assert body["temperature"] == 0.5
    assert body["stream"] is False
    assert body["reasoning_effort"] == "none"          # 关闭思考型模型的推理过程
    assert body["options"]["num_ctx"] == lm.default_num_ctx  # 显式 8192


def test_structured_output_sends_json_schema(monkeypatch):
    capture = patch_post(monkeypatch, [_content_body('{"res": "甲"}')], capture={})
    model = make_model()
    assert model.completion("提示", return_type=Echo, caller="ut") == "甲"
    fmt = capture["body"]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Echo"
    assert fmt["json_schema"]["strict"] is True


# ============ 输出清洗与抢救 ============

def test_think_block_is_filtered(monkeypatch):
    patch_post(monkeypatch, [_content_body("<think>推理过程\n多行</think>答案")])
    model = make_model()
    assert model.completion("提示", caller="ut") == "答案"


@pytest.mark.parametrize(
    "content,expected",
    [
        ('{"res": "甲"}', "甲"),
        ('好的，结果如下：{"res": "乙"} 希望有帮助', "乙"),  # 从文本中抢救 JSON
        ('<think>x</think>{"res": "丙"}', "丙"),
        ("纯文本不是JSON", "纯文本不是JSON"),                # 抢救失败 -> 返回原文
        ('{"res": ', '{"res": '),                            # 内嵌 JSON 也坏了 -> 原文
    ],
)
def test_structured_output_tolerances(monkeypatch, content, expected):
    patch_post(monkeypatch, [_content_body(content)])
    model = make_model()
    assert model.completion("提示", return_type=Echo, caller="ut") == expected


# ============ 重试 / callback / failsafe ============

def test_retry_then_success(monkeypatch):
    patch_post(
        monkeypatch,
        [ConnectionError("boom"), ConnectionError("boom"), _content_body("最终")],
    )
    patch_sleep(monkeypatch)
    model = make_model()
    assert model.completion("提示", retry=5, caller="ut") == "最终"
    summary = model.get_summary()["summary"]["ut"]
    # S=最终成功 F=最终无结果 R=单次调用成功（异常重试不计入 F）
    assert summary == "S:1,F:0/R:1"


def test_retry_exhausted_returns_failsafe(monkeypatch):
    patch_post(monkeypatch, [ConnectionError("boom")] * 3)
    patch_sleep(monkeypatch)
    model = make_model()
    out = model.completion("提示", retry=3, failsafe="兜底", caller="ut")
    assert out == "兜底"
    summary = model.get_summary()["summary"]["ut"]
    assert summary == "S:0,F:1/R:0"  # 全部异常：一次最终失败，零次成功调用


def test_callback_transforms_output(monkeypatch):
    patch_post(monkeypatch, [_content_body("  有空格  ")])
    model = make_model()
    out = model.completion("提示", callback=lambda s: s.strip(), caller="ut")
    assert out == "有空格"


def test_callback_returning_none_triggers_retry(monkeypatch):
    patch_post(monkeypatch, [_content_body("x")] * 3)
    patch_sleep(monkeypatch)
    model = make_model()
    out = model.completion(
        "提示", retry=3, callback=lambda s: None, failsafe="兜底", caller="ut"
    )
    assert out == "兜底"
    summary = model.get_summary()["summary"]["ut"]
    assert summary == "S:0,F:1/R:3"  # 三次调用都成功，但 callback 均返回 None


def test_no_retry_when_success(monkeypatch):
    capture = patch_post(monkeypatch, [_content_body("一次就好")], capture={})
    model = make_model()
    assert model.completion("提示", retry=10, caller="ut") == "一次就好"


# ============ 开关与工厂 ============

def test_disable():
    model = make_model()
    assert model.is_available()
    model.disable()
    assert not model.is_available()


def test_create_llm_model_factory():
    model = lm.create_llm_model(
        {"provider": "ollama", "model": "m", "base_url": "u", "api_key": ""}
    )
    assert isinstance(model, lm.OllamaLLMModel)
    with pytest.raises(NotImplementedError):
        lm.create_llm_model({"provider": "nope", "model": "m", "base_url": "u", "api_key": ""})
