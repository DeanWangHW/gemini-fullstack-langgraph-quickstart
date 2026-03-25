"""提示词构建工具模块。

该模块将“模板字符串”和“渲染逻辑”分离，避免 `graph.py` 出现大量 `format`
细节代码，便于未来升级为多模板/多语言/多场景提示词版本管理。
"""

from __future__ import annotations

from datetime import datetime

from agent.prompts import (
    answer_instructions,
    query_writer_instructions,
    reflection_instructions,
)


def get_current_date() -> str:
    """获取当前日期字符串（英文月份格式）。

    Returns
    -------
    str
        形如 ``"March 25, 2026"`` 的日期字符串。
    """
    return datetime.now().strftime("%B %d, %Y")


def build_query_prompt(
    *,
    research_topic: str,
    number_queries: int,
    current_date: str | None = None,
) -> str:
    """构建查询生成节点提示词。

    Parameters
    ----------
    research_topic : str
        用户问题或对话上下文提炼后的研究主题。
    number_queries : int
        允许模型生成的初始查询数量上限。
    current_date : str or None, optional
        外部注入日期，便于测试稳定性；为空时自动取当前日期。

    Returns
    -------
    str
        渲染后的完整提示词文本。
    """
    return query_writer_instructions.format(
        current_date=current_date or get_current_date(),
        research_topic=research_topic,
        number_queries=number_queries,
    )


def build_reflection_prompt(
    *,
    research_topic: str,
    summaries: list[str],
    current_date: str | None = None,
) -> str:
    """构建反思节点提示词。

    Parameters
    ----------
    research_topic : str
        研究主题。
    summaries : list[str]
        当前已收集的检索摘要列表。
    current_date : str or None, optional
        外部注入日期，便于测试稳定性；为空时自动取当前日期。

    Returns
    -------
    str
        渲染后的完整提示词文本。
    """
    return reflection_instructions.format(
        current_date=current_date or get_current_date(),
        research_topic=research_topic,
        summaries="\n\n---\n\n".join(summaries),
    )


def build_answer_prompt(
    *,
    research_topic: str,
    summaries: list[str],
    current_date: str | None = None,
) -> str:
    """构建最终回答节点提示词。

    Parameters
    ----------
    research_topic : str
        研究主题。
    summaries : list[str]
        当前已收集的检索摘要列表。
    current_date : str or None, optional
        外部注入日期，便于测试稳定性；为空时自动取当前日期。

    Returns
    -------
    str
        渲染后的完整提示词文本。
    """
    return answer_instructions.format(
        current_date=current_date or get_current_date(),
        research_topic=research_topic,
        summaries="\n---\n\n".join(summaries),
    )
