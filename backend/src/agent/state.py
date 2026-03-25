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
    """`generate_query` 节点输入。"""

    messages: list[AnyMessage]
    initial_search_query_count: NotRequired[int]
    reasoning_model: NotRequired[str]


class GenerateQueryOutput(TypedDict):
    """`generate_query` 节点输出。"""

    search_query: list[str]
    reasoning_model: NotRequired[str]


class WebResearchTask(TypedDict):
    """`web_research` 节点任务载荷。"""

    search_query: str
    id: int
    reasoning_model: NotRequired[str]


class WebResearchOutput(TypedDict):
    """`web_research` 节点输出。"""

    sources_gathered: list[SourceRecord]
    search_query: list[str]
    web_research_result: list[str]


class ReflectionInput(TypedDict):
    """`reflection` 节点输入。"""

    messages: list[AnyMessage]
    web_research_result: list[str]
    search_query: list[str]
    research_loop_count: NotRequired[int]
    reasoning_model: NotRequired[str]


class ReflectionOutput(TypedDict):
    """`reflection` 节点输出。"""

    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: list[str]
    research_loop_count: int
    number_of_ran_queries: int
    reasoning_model: NotRequired[str]


class FinalizeAnswerInput(TypedDict):
    """`finalize_answer` 节点输入。"""

    messages: list[AnyMessage]
    web_research_result: list[str]
    sources_gathered: list[SourceRecord]
    reasoning_model: NotRequired[str]


class FinalizeAnswerOutput(TypedDict):
    """`finalize_answer` 节点输出。"""

    messages: list[AIMessage]
    sources_gathered: list[SourceRecord]


# Backward-compatible aliases used by existing code/tests.
OverallState = AgentState
QueryGenerationState = GenerateQueryOutput
WebSearchState = WebResearchTask
ReflectionState = ReflectionOutput


@dataclass(kw_only=True)
class SearchStateOutput:
    """兼容旧接口的搜索结果封装。"""

    running_summary: str = field(default=None)  # Final report
