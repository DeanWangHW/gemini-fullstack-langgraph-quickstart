import sys
from types import SimpleNamespace

import pytest

import agent.llm_client as llm_client_module
from agent.llm_client import LLMClient
from agent.tools_and_schemas import SearchQueryList


def test_parse_json_payload_supports_fenced_json() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert LLMClient._parse_json_payload(raw) == {"a": 1}


def test_make_source_label_prefers_title() -> None:
    assert LLMClient._make_source_label("A title", "https://example.com") == "A title"


def test_make_source_label_uses_hostname_without_title() -> None:
    assert (
        LLMClient._make_source_label("", "https://www.example.com/path")
        == "example.com"
    )


def test_fallback_summary_contains_links() -> None:
    summary = LLMClient._fallback_summary(
        [{"snippet": "Fact 1", "title": "T1", "url": "https://a.com"}],
        [{"label": "a", "short_url": "https://short/1", "value": "https://a.com"}],
    )
    assert "Fact 1" in summary
    assert "https://short/1" in summary


def test_get_client_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        client._get_client()


def test_get_client_passes_base_url_to_openai(monkeypatch) -> None:
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
