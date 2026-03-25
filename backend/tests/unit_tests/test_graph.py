"""`agent.graph` 主流程单元测试。

本文件覆盖图中四个核心节点与路由函数：

1. `generate_query`：结构化查询生成与模型选择；
2. `web_research`：检索结果映射到状态；
3. `reflection` / `evaluate_research`：循环控制与分支决策；
4. `finalize_answer`：引用替换与来源裁剪。
"""

import importlib

from langchain_core.messages import HumanMessage
from langgraph.types import Send

graph_module = importlib.import_module("agent.graph")
from agent.tools_and_schemas import Reflection, SearchQueryList


def test_generate_query_uses_structured_llm(monkeypatch) -> None:
    """验证查询生成节点调用结构化输出接口。"""
    captured = {}

    def fake_generate_structured(prompt, *, schema, model, temperature):
        captured["prompt"] = prompt
        captured["schema"] = schema
        captured["model"] = model
        captured["temperature"] = temperature
        return SearchQueryList(query=["q1", "q2"], rationale="test")

    monkeypatch.setattr(graph_module.llm_client, "generate_structured", fake_generate_structured)

    state = {"messages": [HumanMessage(content="Tell me about LangGraph")]}
    result = graph_module.generate_query(
        state,
        {
            "configurable": {
                "number_of_initial_queries": 2,
                "query_generator_model": "gpt-4.1-mini",
            }
        },
    )

    assert result["search_query"] == ["q1", "q2"]
    assert captured["schema"] is SearchQueryList
    assert captured["model"] == "gpt-4.1-mini"


def test_generate_query_prefers_reasoning_model_from_state(monkeypatch) -> None:
    """验证查询生成节点优先使用请求级模型覆盖。"""
    captured = {}

    def fake_generate_structured(prompt, *, schema, model, temperature):
        captured["model"] = model
        return SearchQueryList(query=["q1"], rationale="test")

    monkeypatch.setattr(
        graph_module.llm_client,
        "generate_structured",
        fake_generate_structured,
    )

    state = {
        "messages": [HumanMessage(content="Tell me about LangGraph")],
        "reasoning_model": "gpt-5",
    }
    graph_module.generate_query(
        state,
        {"configurable": {"query_generator_model": "gpt-4.1-mini"}},
    )
    assert captured["model"] == "gpt-5"


def test_continue_to_web_research_creates_send_objects() -> None:
    """验证查询列表会扇出为多个 `Send` 任务。"""
    sends = graph_module.continue_to_web_research({"search_query": ["a", "b"]})
    assert len(sends) == 2
    assert all(isinstance(item, Send) for item in sends)
    assert sends[0].node == "web_research"
    assert sends[0].arg == {"search_query": "a", "id": 0}
    assert sends[1].arg == {"search_query": "b", "id": 1}


def test_continue_to_web_research_propagates_reasoning_model() -> None:
    """验证 `continue_to_web_research` 会透传模型字段。"""
    sends = graph_module.continue_to_web_research(
        {
            "search_query": ["a", "b"],
            "reasoning_model": "gpt-4.1",
        }
    )
    assert sends[0].arg == {"search_query": "a", "id": 0, "reasoning_model": "gpt-4.1"}
    assert sends[1].arg == {"search_query": "b", "id": 1, "reasoning_model": "gpt-4.1"}


def test_web_research_maps_results_to_state(monkeypatch) -> None:
    """验证检索节点将摘要和来源映射为状态增量。"""
    captured = {}

    def fake_search_and_summarize(**kwargs):
        captured["model"] = kwargs["model"]
        return (
            "summary [src](https://websearch.local/id/1-0)",
            [
                {
                    "label": "src",
                    "short_url": "https://websearch.local/id/1-0",
                    "value": "https://example.com",
                }
            ],
        )

    monkeypatch.setattr(
        graph_module.llm_client,
        "search_and_summarize",
        fake_search_and_summarize,
    )

    result = graph_module.web_research(
        {"search_query": "topic", "id": 1, "reasoning_model": "gpt-5"},
        {"configurable": {"query_generator_model": "gpt-4.1-mini"}},
    )
    assert result["search_query"] == ["topic"]
    assert result["web_research_result"] == ["summary [src](https://websearch.local/id/1-0)"]
    assert len(result["sources_gathered"]) == 1
    assert captured["model"] == "gpt-5"


def test_reflection_increments_loop_and_returns_decision(monkeypatch) -> None:
    """验证反思节点会递增循环计数并返回判定结果。"""
    monkeypatch.setattr(
        graph_module.llm_client,
        "generate_structured",
        lambda *args, **kwargs: Reflection(
            is_sufficient=False,
            knowledge_gap="Need pricing details",
            follow_up_queries=["LangGraph pricing latest"],
        ),
    )

    state = {
        "messages": [HumanMessage(content="What is LangGraph pricing?")],
        "web_research_result": ["partial summary"],
        "search_query": ["initial query"],
    }
    result = graph_module.reflection(state, {"configurable": {"reflection_model": "gpt-4.1-mini"}})
    assert result["is_sufficient"] is False
    assert result["knowledge_gap"] == "Need pricing details"
    assert result["follow_up_queries"] == ["LangGraph pricing latest"]
    assert result["research_loop_count"] == 1
    assert result["number_of_ran_queries"] == 1


def test_evaluate_research_returns_finalize_when_sufficient() -> None:
    """信息充分时应直接路由到最终回答节点。"""
    next_step = graph_module.evaluate_research(
        {"is_sufficient": True, "research_loop_count": 1},
        {"configurable": {"max_research_loops": 3}},
    )
    assert next_step == "finalize_answer"


def test_evaluate_research_returns_follow_up_sends() -> None:
    """信息不足时应生成后续检索分支任务。"""
    next_step = graph_module.evaluate_research(
        {
            "is_sufficient": False,
            "research_loop_count": 1,
            "follow_up_queries": ["q1", "q2"],
            "number_of_ran_queries": 3,
        },
        {"configurable": {"max_research_loops": 3}},
    )
    assert isinstance(next_step, list)
    assert next_step[0].node == "web_research"
    assert next_step[0].arg == {"search_query": "q1", "id": 3}
    assert next_step[1].arg == {"search_query": "q2", "id": 4}


def test_evaluate_research_propagates_reasoning_model_to_followups() -> None:
    """后续检索任务应继承请求级模型字段。"""
    next_step = graph_module.evaluate_research(
        {
            "is_sufficient": False,
            "research_loop_count": 1,
            "follow_up_queries": ["q1", "q2"],
            "number_of_ran_queries": 3,
            "reasoning_model": "gpt-4.1",
        },
        {"configurable": {"max_research_loops": 3}},
    )

    assert isinstance(next_step, list)
    assert next_step[0].arg == {"search_query": "q1", "id": 3, "reasoning_model": "gpt-4.1"}
    assert next_step[1].arg == {"search_query": "q2", "id": 4, "reasoning_model": "gpt-4.1"}


def test_finalize_answer_replaces_short_urls(monkeypatch) -> None:
    """最终回答应将短链接替换为真实 URL，并裁剪未引用来源。"""
    monkeypatch.setattr(
        graph_module.llm_client,
        "generate_text",
        lambda *args, **kwargs: "Final answer uses [src](https://websearch.local/id/9-0)",
    )
    state = {
        "messages": [HumanMessage(content="question")],
        "web_research_result": ["summary"],
        "sources_gathered": [
            {
                "label": "src",
                "short_url": "https://websearch.local/id/9-0",
                "value": "https://example.com/source",
            },
            {
                "label": "unused",
                "short_url": "https://websearch.local/id/9-1",
                "value": "https://example.com/unused",
            },
        ],
    }
    result = graph_module.finalize_answer(
        state,
        {"configurable": {"answer_model": "gpt-4.1"}},
    )
    output_text = result["messages"][0].content
    assert "https://example.com/source" in output_text
    assert "https://websearch.local/id/9-0" not in output_text
    assert result["sources_gathered"] == [
        {
            "label": "src",
            "short_url": "https://websearch.local/id/9-0",
            "value": "https://example.com/source",
        }
    ]
