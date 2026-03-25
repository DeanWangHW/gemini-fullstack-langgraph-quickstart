"""模型选择策略模块。

该模块统一封装“节点使用哪个模型”的决策逻辑，避免在 `graph.py` 中散落
重复分支判断，便于后续按阶段扩展（例如按节点维度加路由策略、按任务复杂度
动态切换模型等）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.configuration import Configuration


def select_query_model(state: Mapping[str, Any], config: Configuration) -> str:
    """选择查询生成阶段模型。

    Parameters
    ----------
    state : Mapping[str, Any]
        当前图状态。
    config : Configuration
        解析后的全局配置。

    Returns
    -------
    str
        优先返回 `state.reasoning_model`，否则回退到 `query_generator_model`。
    """
    return state.get("reasoning_model") or config.query_generator_model


def select_research_model(state: Mapping[str, Any], config: Configuration) -> str:
    """选择网页检索摘要阶段模型。

    Parameters
    ----------
    state : Mapping[str, Any]
        当前图状态。
    config : Configuration
        解析后的全局配置。

    Returns
    -------
    str
        优先返回 `state.reasoning_model`，否则回退到 `query_generator_model`。
    """
    return state.get("reasoning_model") or config.query_generator_model


def select_reflection_model(state: Mapping[str, Any], config: Configuration) -> str:
    """选择反思阶段模型。

    Parameters
    ----------
    state : Mapping[str, Any]
        当前图状态。
    config : Configuration
        解析后的全局配置。

    Returns
    -------
    str
        优先返回 `state.reasoning_model`，否则回退到 `reflection_model`。
    """
    return state.get("reasoning_model") or config.reflection_model


def select_answer_model(state: Mapping[str, Any], config: Configuration) -> str:
    """选择最终回答阶段模型。

    Parameters
    ----------
    state : Mapping[str, Any]
        当前图状态。
    config : Configuration
        解析后的全局配置。

    Returns
    -------
    str
        优先返回 `state.reasoning_model`，否则回退到 `answer_model`。
    """
    return state.get("reasoning_model") or config.answer_model


def carry_reasoning_model(state: Mapping[str, Any]) -> dict[str, str]:
    """在分支跳转时透传 `reasoning_model`。

    Parameters
    ----------
    state : Mapping[str, Any]
        当前图状态或节点输出。

    Returns
    -------
    dict[str, str]
        若存在 `reasoning_model`，返回 `{"reasoning_model": value}`；
        否则返回空字典。
    """
    reasoning_model = state.get("reasoning_model")
    if not reasoning_model:
        return {}
    return {"reasoning_model": str(reasoning_model)}
