"""LangGraph 主流程定义模块。

状态流固定为：

`generate_query -> web_research -> reflection -> finalize_answer`

其中：

1. `generate_query` 负责生成检索查询；
2. `web_research` 负责执行检索并摘要；
3. `reflection` 判断是否需要继续检索；
4. `finalize_answer` 汇总并替换最终引用链接。
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.configuration import Configuration
from agent.llm_client import llm_client
from agent.model_selection import (
    carry_reasoning_model,
    select_answer_model,
    select_query_model,
    select_reflection_model,
    select_research_model,
)
from agent.prompt_builder import (
    build_answer_prompt,
    build_query_prompt,
    build_reflection_prompt,
)
from agent.state import (
    AgentState,
    FinalizeAnswerInput,
    FinalizeAnswerOutput,
    GenerateQueryInput,
    GenerateQueryOutput,
    ReflectionOutput,
    WebResearchOutput,
    WebResearchTask,
)
from agent.tools_and_schemas import Reflection, SearchQueryList
from agent.utils import get_research_topic

load_dotenv()


# Nodes
def generate_query(
    state: GenerateQueryInput,
    config: RunnableConfig,
) -> GenerateQueryOutput:
    """生成初始检索查询列表。

    Parameters
    ----------
    state : GenerateQueryInput
        当前节点输入，至少包含用户消息列表。
    config : RunnableConfig
        LangGraph 运行时配置，用于解析模型与循环参数。

    Returns
    -------
    GenerateQueryOutput
        包含 `search_query` 的结构化结果，并在可用时透传 `reasoning_model`。

    Notes
    -----
    - 查询词生成使用结构化输出 `SearchQueryList`，减少解析不确定性。
    - 节点不做任何外部副作用，仅返回状态增量。
    """
    configurable = Configuration.from_runnable_config(config)
    query_model = select_query_model(state, configurable)
    number_queries = state.get(
        "initial_search_query_count", configurable.number_of_initial_queries
    )
    formatted_prompt = build_query_prompt(
        research_topic=get_research_topic(state["messages"]),
        number_queries=number_queries,
    )
    result = llm_client.generate_structured(
        formatted_prompt,
        schema=SearchQueryList,
        model=query_model,
        temperature=1.0,
    )
    output: GenerateQueryOutput = {"search_query": result.query}
    output.update(carry_reasoning_model(state))
    return output


def continue_to_web_research(state: GenerateQueryOutput) -> list[Send]:
    """把查询列表扇出为多个 `web_research` 任务。

    Parameters
    ----------
    state : GenerateQueryOutput
        查询生成节点输出。

    Returns
    -------
    list[Send]
        每条查询对应一个 `Send("web_research", payload)`。

    Notes
    -----
    - 任务 ID 使用查询索引构造，保证同一批次内唯一。
    - 若上游指定 `reasoning_model`，会被透传到每个分支任务。
    """
    sends: list[Send] = []
    reasoning_payload = carry_reasoning_model(state)
    for idx, search_query in enumerate(state["search_query"]):
        payload: dict[str, int | str] = {
            "search_query": search_query,
            "id": int(idx),
        }
        payload.update(reasoning_payload)
        sends.append(Send("web_research", payload))
    return sends


def web_research(
    state: WebResearchTask,
    config: RunnableConfig,
) -> WebResearchOutput:
    """执行单条查询的网页检索与摘要。

    Parameters
    ----------
    state : WebResearchTask
        检索任务载荷，包含查询文本与任务 ID。
    config : RunnableConfig
        运行时配置对象。

    Returns
    -------
    WebResearchOutput
        检索输出增量，包含：

        - `sources_gathered`：来源列表；
        - `search_query`：本次已执行查询（列表形式，便于 reducer 追加）；
        - `web_research_result`：摘要文本列表。

    Notes
    -----
    检索细节（DDG backend 重试、结果归一化、引用占位符）由 `llm_client`
    内部负责，图节点只做编排与状态拼装。
    """
    configurable = Configuration.from_runnable_config(config)
    research_model = select_research_model(state, configurable)
    summary_text, sources_gathered = llm_client.search_and_summarize(
        query=state["search_query"],
        query_id=int(state["id"]),
        model=research_model,
    )

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [summary_text],
    }


def reflection(state: AgentState, config: RunnableConfig) -> ReflectionOutput:
    """反思当前检索结果并生成下一轮查询建议。

    Parameters
    ----------
    state : AgentState
        当前全局状态。
    config : RunnableConfig
        运行时配置对象。

    Returns
    -------
    ReflectionOutput
        反思节点输出，包含信息充分性判定、知识缺口说明和后续查询列表。

    Notes
    -----
    - `research_loop_count` 在此节点递增；
    - 结构化输出由 `Reflection` schema 强约束，降低 JSON 解析失败概率。
    """
    configurable = Configuration.from_runnable_config(config)
    research_loop_count = state.get("research_loop_count", 0) + 1
    reasoning_model = select_reflection_model(state, configurable)
    formatted_prompt = build_reflection_prompt(
        research_topic=get_research_topic(state["messages"]),
        summaries=state["web_research_result"],
    )
    result = llm_client.generate_structured(
        formatted_prompt,
        schema=Reflection,
        model=reasoning_model,
        temperature=1.0,
    )

    output: ReflectionOutput = {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": research_loop_count,
        "number_of_ran_queries": len(state["search_query"]),
    }
    output.update(carry_reasoning_model(state))
    return output


def evaluate_research(
    state: AgentState,
    config: RunnableConfig,
) -> str | list[Send]:
    """根据反思结果决定流程走向。

    Parameters
    ----------
    state : AgentState
        当前全局状态（需包含反思节点输出字段）。
    config : RunnableConfig
        运行时配置对象。

    Returns
    -------
    str or list[Send]
        - 若信息充分或达最大循环次数，返回 `"finalize_answer"`；
        - 否则返回 follow-up 查询对应的 `Send` 列表。

    Notes
    -----
    该函数是图中的关键路由节点，直接决定“继续检索”还是“结束总结”。
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        sends: list[Send] = []
        reasoning_payload = carry_reasoning_model(state)
        for idx, follow_up_query in enumerate(state["follow_up_queries"]):
            payload: dict[str, int | str] = {
                "search_query": follow_up_query,
                "id": state["number_of_ran_queries"] + int(idx),
            }
            payload.update(reasoning_payload)
            sends.append(Send("web_research", payload))
        return sends


def finalize_answer(
    state: FinalizeAnswerInput,
    config: RunnableConfig,
) -> FinalizeAnswerOutput:
    """合成最终回答并替换短链接为真实来源链接。

    Parameters
    ----------
    state : FinalizeAnswerInput
        最终节点输入，包含消息、摘要和来源集合。
    config : RunnableConfig
        运行时配置对象。

    Returns
    -------
    FinalizeAnswerOutput
        包含：

        - `messages`：最终 AI 回复；
        - `sources_gathered`：仅保留最终答案实际引用到的来源。

    Notes
    -----
    节点会遍历全部来源占位符，并把正文中的短链接替换为真实 URL，
    以便前端展示与后续持久化时都使用可访问链接。
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = select_answer_model(state, configurable)
    formatted_prompt = build_answer_prompt(
        research_topic=get_research_topic(state["messages"]),
        summaries=state["web_research_result"],
    )

    final_answer_text = llm_client.generate_text(
        formatted_prompt,
        model=reasoning_model,
        temperature=0,
    )

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in final_answer_text:
            final_answer_text = final_answer_text.replace(
                source["short_url"],
                source["value"],
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=final_answer_text)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(AgentState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
