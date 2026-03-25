"""`agent.llm_client` 模块单元测试。

本文件重点验证以下能力：

1. JSON 解析、来源标签与兜底摘要等纯函数行为；
2. OpenAI 客户端初始化参数与异常路径；
3. DDG 检索重试、backend 兼容与查询清洗逻辑；
4. 检索失败时的错误可读性与结果稳定性。
"""

import sys
from types import SimpleNamespace

import pytest

import agent.llm_client as llm_client_module
from agent.llm_client import LLMClient
from agent.tools_and_schemas import SearchQueryList


def test_parse_json_payload_supports_fenced_json() -> None:
    """应支持解析带 ```json 围栏的模型输出。"""
    raw = '```json\n{"a": 1}\n```'
    assert LLMClient._parse_json_payload(raw) == {"a": 1}


def test_make_source_label_prefers_title() -> None:
    """有标题时应优先使用标题作为来源标签。"""
    assert LLMClient._make_source_label("A title", "https://example.com") == "A title"


def test_make_source_label_uses_hostname_without_title() -> None:
    """标题缺失时应退化为域名标签。"""
    assert (
        LLMClient._make_source_label("", "https://www.example.com/path")
        == "example.com"
    )


def test_fallback_summary_contains_links() -> None:
    """兜底摘要应包含文本事实与短链接。"""
    summary = LLMClient._fallback_summary(
        [{"snippet": "Fact 1", "title": "T1", "url": "https://a.com"}],
        [{"label": "a", "short_url": "https://short/1", "value": "https://a.com"}],
    )
    assert "Fact 1" in summary
    assert "https://short/1" in summary


def test_get_client_requires_api_key(monkeypatch) -> None:
    """缺失 API Key 时应抛出明确异常。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        client._get_client()


def test_get_client_passes_base_url_to_openai(monkeypatch) -> None:
    """初始化 OpenAI 客户端时应正确传递 `base_url`。"""
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_client_module, "OpenAI", FakeOpenAI)
    client = LLMClient(api_key="test-key", base_url="https://custom.openai.local/v1")
    created = client._get_client()

    assert isinstance(created, FakeOpenAI)
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://custom.openai.local/v1"


def test_generate_structured_parses_schema(monkeypatch) -> None:
    """结构化生成应能通过 schema 校验并返回模型对象。"""
    client = LLMClient(api_key="test-key")
    monkeypatch.setattr(
        client,
        "_chat_completion",
        lambda **kwargs: '{"query":["q1","q2"],"rationale":"because"}',
    )
    result = client.generate_structured(
        "prompt",
        schema=SearchQueryList,
        model="gpt-4.1-mini",
    )
    assert result.query == ["q1", "q2"]
    assert result.rationale == "because"


def test_search_and_summarize_returns_empty_result_message(monkeypatch) -> None:
    """无检索结果时应返回可读提示且来源为空。"""
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=5):
            return []

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    client = LLMClient(api_key="test-key")
    summary, sources = client.search_and_summarize(
        query="nothing",
        query_id=1,
        model="gpt-4.1-mini",
    )
    assert "No search results" in summary
    assert sources == []


def test_search_and_summarize_uses_fallback_when_model_has_no_citations(monkeypatch) -> None:
    """模型摘要无引用时应回退到内置可引用摘要。"""
    search_results = [
        {"title": "First Source", "href": "https://a.com", "body": "Alpha fact"},
        {"title": "Second Source", "href": "https://b.com", "body": "Beta fact"},
    ]

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=5):
            return search_results

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    client = LLMClient(api_key="test-key")
    monkeypatch.setattr(
        client,
        "generate_text",
        lambda prompt, model, temperature=0.0: "summary without markers",
    )

    summary, sources = client.search_and_summarize(
        query="topic",
        query_id=7,
        model="gpt-4.1-mini",
    )
    assert len(sources) == 2
    assert "https://websearch.local/id/7-0" in summary
    assert "https://websearch.local/id/7-1" in summary


def test_search_and_summarize_retries_backends_and_recovers(monkeypatch) -> None:
    """前置 backend 失败后应继续重试并最终恢复。"""
    class FakeDDGS:
        call_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=5, backend=None):
            FakeDDGS.call_count += 1
            if FakeDDGS.call_count < 3:
                raise RuntimeError("temporary backend failure")
            return [{"title": "Recovered", "href": "https://ok.example", "body": "ok"}]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    client = LLMClient(api_key="test-key")
    monkeypatch.setattr(
        client,
        "generate_text",
        lambda prompt, model, temperature=0.0: "Recovered fact [Recovered](https://websearch.local/id/5-0)",
    )

    summary, sources = client.search_and_summarize(
        query="topic",
        query_id=5,
        model="gpt-4.1-mini",
    )
    assert "https://websearch.local/id/5-0" in summary
    assert len(sources) == 1


def test_search_and_summarize_reports_provider_errors_when_all_fail(monkeypatch) -> None:
    """所有 backend 失败时应返回 provider 错误摘要。"""
    class AlwaysFailDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=5, backend=None):
            raise RuntimeError("provider down")

    monkeypatch.setitem(
        sys.modules,
        "ddgs",
        SimpleNamespace(DDGS=AlwaysFailDDGS),
    )

    client = LLMClient(api_key="test-key")
    summary, sources = client.search_and_summarize(
        query="topic",
        query_id=10,
        model="gpt-4.1-mini",
    )
    assert "No search results were found" in summary
    assert "Search provider errors:" in summary
    assert sources == []


def test_search_and_summarize_falls_back_to_default_backend(monkeypatch) -> None:
    """当 provider 不支持 backend 参数时应回退默认调用。"""
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=5, backend=None):
            if backend is not None:
                raise ValueError(f"Unsupported backend {backend}")
            return [{"title": "Recovered", "href": "https://ok.example", "body": "ok"}]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    client = LLMClient(api_key="test-key")
    monkeypatch.setattr(
        client,
        "generate_text",
        lambda prompt, model, temperature=0.0: "Recovered fact [Recovered](https://websearch.local/id/12-0)",
    )

    summary, sources = client.search_and_summarize(
        query="topic",
        query_id=12,
        model="gpt-4.1-mini",
    )
    assert "https://websearch.local/id/12-0" in summary
    assert len(sources) == 1


def test_normalize_search_rows_handles_link_and_description_keys() -> None:
    """归一化逻辑应兼容 `link/description` 字段命名。"""
    rows = LLMClient._normalize_search_rows(
        [
            {
                "title": "Example source",
                "link": "https://example.com",
                "description": "Description text",
            }
        ]
    )

    assert rows == [
        {
            "title": "Example source",
            "url": "https://example.com",
            "snippet": "Description text",
        }
    ]


def test_search_and_summarize_sanitizes_long_query_before_search(monkeypatch) -> None:
    """长自然语言查询应被压缩并移除 citation 指令。"""
    long_query = (
        "What were average global lithium-ion battery pack prices in 2024 and 2025, "
        "the outlook for 2026, and the differences by chemistry (LFP vs NMC) and "
        "application (EV vs stationary)? Please cite the BloombergNEF Battery Price "
        "Survey 2025 and other primary sources."
    )

    class FakeDDGS:
        queries_seen: list[str] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=5, backend=None):
            FakeDDGS.queries_seen.append(query)
            if len(query) > 150 or "Please cite" in query:
                raise RuntimeError("provider rejected long query")
            return [
                {
                    "title": "Battery price report",
                    "href": "https://example.com/battery-prices",
                    "body": "Average battery pack prices declined in 2024.",
                }
            ]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    client = LLMClient(api_key="test-key")
    monkeypatch.setattr(
        client,
        "generate_text",
        lambda prompt, model, temperature=0.0: (
            "Battery prices fell [Battery price report](https://websearch.local/id/31-0)"
        ),
    )

    summary, sources = client.search_and_summarize(
        query=long_query,
        query_id=31,
        model="gpt-4.1-mini",
    )

    assert FakeDDGS.queries_seen
    assert "Please cite" not in FakeDDGS.queries_seen[0]
    assert len(FakeDDGS.queries_seen[0]) <= 150
    assert "https://websearch.local/id/31-0" in summary
    assert len(sources) == 1
