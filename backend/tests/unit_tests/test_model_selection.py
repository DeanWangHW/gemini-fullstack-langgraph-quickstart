"""`agent.model_selection` 模块单元测试。

用于验证各节点模型选择策略的优先级规则：

1. 有 `reasoning_model` 时优先使用；
2. 无覆盖值时回退到节点默认模型；
3. 透传函数仅在字段存在时输出 payload。
"""

from agent.configuration import Configuration
from agent.model_selection import (
    carry_reasoning_model,
    select_answer_model,
    select_query_model,
    select_reflection_model,
    select_research_model,
)


def test_select_query_model_prefers_reasoning_model() -> None:
    """查询节点应优先使用请求级 `reasoning_model`。"""
    cfg = Configuration(query_generator_model="gpt-4.1-mini")
    state = {"reasoning_model": "gpt-4.1"}
    assert select_query_model(state, cfg) == "gpt-4.1"


def test_select_research_model_falls_back_to_query_generator_model() -> None:
    """检索节点在未覆盖时应回退到查询模型。"""
    cfg = Configuration(query_generator_model="gpt-4.1-mini")
    state = {}
    assert select_research_model(state, cfg) == "gpt-4.1-mini"


def test_select_reflection_model_falls_back_to_reflection_model() -> None:
    """反思节点在未覆盖时应使用反思模型默认值。"""
    cfg = Configuration(reflection_model="gpt-4o-mini")
    state = {}
    assert select_reflection_model(state, cfg) == "gpt-4o-mini"


def test_select_answer_model_falls_back_to_answer_model() -> None:
    """最终回答节点在未覆盖时应使用答案模型默认值。"""
    cfg = Configuration(answer_model="gpt-4.1")
    state = {}
    assert select_answer_model(state, cfg) == "gpt-4.1"


def test_carry_reasoning_model_only_when_present() -> None:
    """仅在存在字段时透传 `reasoning_model`。"""
    assert carry_reasoning_model({"reasoning_model": "gpt-4.1"}) == {
        "reasoning_model": "gpt-4.1"
    }
    assert carry_reasoning_model({}) == {}
