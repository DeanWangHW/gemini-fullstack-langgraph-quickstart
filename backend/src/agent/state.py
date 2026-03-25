"""LangGraph 状态类型定义模块。

本模块集中定义图内各节点的输入/输出状态契约，目标是：

1. 让 `generate_query -> web_research -> reflection -> finalize_answer`
   的数据流具有明确、可测试的结构边界；
2. 通过 `TypedDict` + reducer 注解表达“可追加字段”和“节点私有字段”；
3. 为前后端联调、单元测试与后续重构提供稳定的类型基线。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated, NotRequired


class SourceRecord(TypedDict):
    """标准化来源记录结构。

    Attributes
    ----------
    label : str
        来源标题（用于展示）。
    short_url : str
        在中间步骤中使用的短链接占位符。
    value : str
        原始真实 URL。
    """

    label: str
    short_url: str
    value: str


class AgentState(TypedDict, total=False):
    """LangGraph 全局状态定义（主状态容器）。

    Attributes
    ----------
    messages : list[AnyMessage]
        对话消息列表（含用户问题和最终答案）。
    search_query : list[str]
        已经执行过或待执行的查询列表（按 reducer 追加）。
    web_research_result : list[str]
        每次检索步骤输出的摘要文本（按 reducer 追加）。
    sources_gathered : list[SourceRecord]
        所有检索步骤收集到的来源记录（按 reducer 追加）。
    initial_search_query_count : int
        初始查询数量（可由请求覆盖）。
    max_research_loops : int
        最大研究循环次数（可由请求覆盖）。
    research_loop_count : int
        当前已经执行的反思循环计数。
    reasoning_model : str
        请求级别统一推理模型，若存在则覆盖分阶段默认模型。
    is_sufficient : bool
        反思节点给出的“信息是否充分”判定。
    knowledge_gap : str
        当前识别出的知识缺口描述。
    follow_up_queries : list[str]
        用于下一轮补充检索的查询列表。
    number_of_ran_queries : int
        已运行查询数量计数，用于生成后续查询唯一 ID。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    search_query: Annotated[list[str], operator.add]
    web_research_result: Annotated[list[str], operator.add]
    sources_gathered: Annotated[list[SourceRecord], operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: list[str]
    number_of_ran_queries: int


class GenerateQueryInput(TypedDict):
    """`generate_query` 节点输入。

    Attributes
    ----------
    messages : list[AnyMessage]
        对话消息列表，至少包含当前用户问题。
    initial_search_query_count : int, optional
        请求级初始查询数量覆盖值；未提供时走全局配置。
    reasoning_model : str, optional
        请求级统一模型覆盖值；若提供则优先于节点默认模型。
    """

    messages: list[AnyMessage]
    initial_search_query_count: NotRequired[int]
    reasoning_model: NotRequired[str]


class GenerateQueryOutput(TypedDict):
    """`generate_query` 节点输出。

    Attributes
    ----------
    search_query : list[str]
        结构化生成的查询列表，供后续检索节点扇出执行。
    reasoning_model : str, optional
        透传到后续节点的请求级模型覆盖值。
    """

    search_query: list[str]
    reasoning_model: NotRequired[str]


class WebResearchTask(TypedDict):
    """`web_research` 节点任务载荷。

    Attributes
    ----------
    search_query : str
        单条检索查询文本。
    id : int
        查询任务 ID，用于构造引用短链接命名空间。
    reasoning_model : str, optional
        请求级统一模型覆盖值。
    """

    search_query: str
    id: int
    reasoning_model: NotRequired[str]


class WebResearchOutput(TypedDict):
    """`web_research` 节点输出。

    Attributes
    ----------
    sources_gathered : list[SourceRecord]
        本次查询采集到的来源结构列表。
    search_query : list[str]
        已执行查询列表（包装为列表以便 reducer 追加）。
    web_research_result : list[str]
        检索摘要文本列表（单次节点通常仅 1 条）。
    """

    sources_gathered: list[SourceRecord]
    search_query: list[str]
    web_research_result: list[str]


class ReflectionInput(TypedDict):
    """`reflection` 节点输入。

    Attributes
    ----------
    messages : list[AnyMessage]
        用户与助手的历史消息。
    web_research_result : list[str]
        已累计的检索摘要。
    search_query : list[str]
        已执行查询列表。
    research_loop_count : int, optional
        当前循环计数；未提供时视为 0。
    reasoning_model : str, optional
        请求级统一模型覆盖值。
    """

    messages: list[AnyMessage]
    web_research_result: list[str]
    search_query: list[str]
    research_loop_count: NotRequired[int]
    reasoning_model: NotRequired[str]


class ReflectionOutput(TypedDict):
    """`reflection` 节点输出。

    Attributes
    ----------
    is_sufficient : bool
        当前摘要是否足够回答用户问题。
    knowledge_gap : str
        当前仍缺失的关键信息描述。
    follow_up_queries : list[str]
        下一轮建议执行的补充查询列表。
    research_loop_count : int
        自增后的循环次数。
    number_of_ran_queries : int
        截至当前已执行查询总量。
    reasoning_model : str, optional
        透传到后续节点的请求级模型覆盖值。
    """

    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: list[str]
    research_loop_count: int
    number_of_ran_queries: int
    reasoning_model: NotRequired[str]


class FinalizeAnswerInput(TypedDict):
    """`finalize_answer` 节点输入。

    Attributes
    ----------
    messages : list[AnyMessage]
        对话消息列表。
    web_research_result : list[str]
        累计检索摘要。
    sources_gathered : list[SourceRecord]
        累计来源记录（含短链接与真实链接映射）。
    reasoning_model : str, optional
        请求级统一模型覆盖值。
    """

    messages: list[AnyMessage]
    web_research_result: list[str]
    sources_gathered: list[SourceRecord]
    reasoning_model: NotRequired[str]


class FinalizeAnswerOutput(TypedDict):
    """`finalize_answer` 节点输出。

    Attributes
    ----------
    messages : list[AIMessage]
        最终答案消息列表（通常仅一条 AIMessage）。
    sources_gathered : list[SourceRecord]
        仅保留最终答案实际引用到的来源集合。
    """

    messages: list[AIMessage]
    sources_gathered: list[SourceRecord]


# Backward-compatible aliases used by existing code/tests.
OverallState = AgentState
QueryGenerationState = GenerateQueryOutput
WebSearchState = WebResearchTask
ReflectionState = ReflectionOutput


@dataclass(kw_only=True)
class SearchStateOutput:
    """兼容旧接口的搜索结果封装。

    Attributes
    ----------
    running_summary : str | None
        旧版本对外暴露的“运行中摘要”字段。当前主流程已不依赖该字段，
        但为兼容历史调用方仍保留数据结构定义。
    """

    running_summary: str = field(default=None)  # Final report
