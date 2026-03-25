"""`agent.utils` 模块单元测试。

覆盖研究主题拼接、短链接映射、引用标记插入与旧版 grounding 引用解析，
用于保证工具函数在历史兼容与当前流程中行为一致。
"""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)


def test_get_research_topic_single_message() -> None:
    """单轮对话应直接返回用户问题文本。"""
    messages = [HumanMessage(content="Explain LangGraph")]
    assert get_research_topic(messages) == "Explain LangGraph"


def test_get_research_topic_with_history() -> None:
    """多轮对话应按 User/Assistant 前缀展开上下文。"""
    messages = [
        HumanMessage(content="Question 1"),
        AIMessage(content="Answer 1"),
        HumanMessage(content="Question 2"),
    ]
    output = get_research_topic(messages)
    assert "User: Question 1" in output
    assert "Assistant: Answer 1" in output
    assert "User: Question 2" in output


def test_resolve_urls_deduplicates_entries() -> None:
    """相同 URL 应映射为同一短链接，避免重复来源。"""
    chunks = [
        SimpleNamespace(web=SimpleNamespace(uri="https://example.com/a")),
        SimpleNamespace(web=SimpleNamespace(uri="https://example.com/b")),
        SimpleNamespace(web=SimpleNamespace(uri="https://example.com/a")),
    ]
    resolved = resolve_urls(chunks, id=3)
    assert len(resolved) == 2
    assert resolved["https://example.com/a"] == "https://vertexaisearch.cloud.google.com/id/3-0"
    assert resolved["https://example.com/b"] == "https://vertexaisearch.cloud.google.com/id/3-1"


def test_insert_citation_markers_appends_expected_markers() -> None:
    """引用标记应插入到对应文本区间末尾。"""
    text = "Alpha Beta Gamma"
    citations = [
        {
            "start_index": 0,
            "end_index": 5,
            "segments": [{"label": "a", "short_url": "https://s/1"}],
        },
        {
            "start_index": 6,
            "end_index": 10,
            "segments": [{"label": "b", "short_url": "https://s/2"}],
        },
    ]
    output = insert_citation_markers(text, citations)
    assert "[a](https://s/1)" in output
    assert "[b](https://s/2)" in output


def test_get_citations_extracts_segments() -> None:
    """应从 grounding 结构中提取区间与来源段信息。"""
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                grounding_metadata=SimpleNamespace(
                    grounding_chunks=[
                        SimpleNamespace(
                            web=SimpleNamespace(
                                uri="https://example.com/a",
                                title="example.com",
                            )
                        )
                    ],
                    grounding_supports=[
                        SimpleNamespace(
                            segment=SimpleNamespace(start_index=0, end_index=6),
                            grounding_chunk_indices=[0],
                        )
                    ],
                )
            )
        ]
    )

    citations = get_citations(
        response,
        {"https://example.com/a": "https://vertexaisearch.cloud.google.com/id/1-0"},
    )
    assert len(citations) == 1
    assert citations[0]["start_index"] == 0
    assert citations[0]["end_index"] == 6
    assert citations[0]["segments"][0]["label"] == "example"
