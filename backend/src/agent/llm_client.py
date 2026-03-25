import json
import os
import re
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Explicitly load backend/.env so variable resolution does not depend on cwd.
BACKEND_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=BACKEND_ENV_PATH)


class LLMClient:
    """Wrap OpenAI model calls and web search utilities."""

    _SEARCH_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "including",
        "is",
        "latest",
        "of",
        "on",
        "or",
        "please",
        "the",
        "to",
        "what",
        "were",
        "with",
        "vs",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.client: OpenAI | None = None

    def generate_text(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
    ) -> str:
        """Generate plain text from an OpenAI chat model."""
        return self._chat_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
        )

    def generate_structured(
        self,
        prompt: str,
        *,
        schema: type[T],
        model: str,
        temperature: float = 0.0,
    ) -> T:
        """Generate JSON and validate it against a Pydantic schema."""
        raw_text = self._chat_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        payload = self._parse_json_payload(raw_text)
        return schema.model_validate(payload)

    def search_and_summarize(
        self,
        *,
        query: str,
        query_id: int,
        model: str,
        max_results: int = 5,
    ) -> tuple[str, list[dict[str, str]]]:
        """Run web search and summarize findings with source markers."""
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise ImportError(
                "ddgs is required for web research. "
                "Install backend dependencies again."
            ) from exc

        normalized_query = self._prepare_search_query(query)
        raw_results, search_errors = self._collect_search_results(
            DDGS=DDGS,
            query=normalized_query,
            max_results=max_results,
        )

        deduped_results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for result in raw_results:
            if result["url"] in seen_urls:
                continue
            seen_urls.add(result["url"])
            deduped_results.append(result)

        if not deduped_results:
            error_context = ""
            if search_errors:
                error_context = (
                    " Search provider errors: " + " | ".join(search_errors[:2])
                )
            return (
                f"No search results were found for '{query}'.{error_context}",
                [],
            )

        limited_results = deduped_results[:max_results]
        sources: list[dict[str, str]] = []
        source_lines: list[str] = []

        for idx, result in enumerate(limited_results):
            short_url = f"https://websearch.local/id/{query_id}-{idx}"
            label = self._make_source_label(result["title"], result["url"])
            source = {
                "label": label,
                "short_url": short_url,
                "value": result["url"],
            }
            sources.append(source)
            source_lines.append(
                "\n".join(
                    [
                        f"- [{label}]({short_url})",
                        f"  Title: {result['title'] or 'Untitled'}",
                        f"  Snippet: {result['snippet'] or 'N/A'}",
                    ]
                )
            )

        summarize_prompt = (
            "You are preparing research notes from web search snippets.\n"
            f"User query: {query}\n\n"
            "Use only the source snippets below.\n"
            "Write 5-8 concise bullet points with factual findings.\n"
            "Every bullet must include at least one citation link copied exactly "
            "from the source list markdown links.\n"
            "Do not invent facts beyond the snippets.\n\n"
            "Sources:\n"
            f"{chr(10).join(source_lines)}\n"
        )

        summary = self.generate_text(
            summarize_prompt,
            model=model,
            temperature=0.2,
        )

        if not any(source["short_url"] in summary for source in sources):
            summary = self._fallback_summary(limited_results, sources)

        return summary, sources

    @classmethod
    def _prepare_search_query(cls, query: str) -> str:
        """Normalize model-generated search queries before sending to DDG."""
        normalized = " ".join((query or "").split()).strip()
        if not normalized:
            return ""

        without_citation = re.split(
            r"\b(?:please\s+cite|cite(?:\s+the)?(?:\s+latest)?)\b",
            normalized,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,.;:")
        prepared = without_citation or normalized

        if len(prepared) > 160 or len(prepared.split()) > 20:
            return cls._simplify_search_query(prepared)

        return prepared

    @classmethod
    def _simplify_search_query(cls, query: str, max_terms: int = 18) -> str:
        """Convert long natural-language prompts into compact keyword queries."""
        normalized = re.sub(r"[–—]", " ", query)
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.+\-/]*", normalized)

        if not tokens:
            return cls._truncate_query(query, max_length=160)

        keywords: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            key = token.casefold().strip(".")
            if not key:
                continue
            if key in cls._SEARCH_STOPWORDS:
                continue
            if len(key) == 1 and not key.isdigit():
                continue
            if key in seen:
                continue
            seen.add(key)
            keywords.append(token.strip("."))
            if len(keywords) >= max_terms:
                break

        if not keywords:
            return cls._truncate_query(query, max_length=160)

        return cls._truncate_query(" ".join(keywords), max_length=160)

    def _collect_search_results(
        self,
        *,
        DDGS,
        query: str,
        max_results: int,
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Run DDG queries with backend fallbacks and collect normalized results."""
        backends: list[str | None] = [None, "html", "lite", "bing"]
        errors: list[str] = []

        for backend in backends:
            backend_label = backend or "default"
            try:
                with DDGS() as ddgs:
                    iterator = self._run_ddgs_text(
                        ddgs=ddgs,
                        query=query,
                        max_results=max_results,
                        backend=backend,
                    )
                    rows = self._normalize_search_rows(iterator)
                    if rows:
                        return rows, errors
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{backend_label}: {type(exc).__name__}: {exc}")

        return [], errors

    @staticmethod
    def _normalize_search_rows(items: Any) -> list[dict[str, str]]:
        """Normalize DDG rows into title/url/snippet records."""
        normalized: list[dict[str, str]] = []
        if items is None:
            return normalized

        iterable = [items] if isinstance(items, dict) else items

        for item in iterable:
            if not isinstance(item, dict):
                continue
            url = item.get("href") or item.get("url") or item.get("link")
            if not isinstance(url, str) or not url.strip():
                continue

            title = item.get("title") or item.get("name") or ""
            snippet = (
                item.get("body")
                or item.get("snippet")
                or item.get("description")
                or item.get("content")
                or ""
            )

            normalized.append(
                {
                    "title": str(title).strip(),
                    "url": url.strip(),
                    "snippet": str(snippet).strip(),
                }
            )
        return normalized

    @staticmethod
    def _run_ddgs_text(
        *,
        ddgs: Any,
        query: str,
        max_results: int,
        backend: str | None,
    ) -> Any:
        """Call DDGS text search with compatibility fallback for old signatures."""
        if backend is None:
            return ddgs.text(query, max_results=max_results)
        try:
            return ddgs.text(
                query,
                max_results=max_results,
                backend=backend,
            )
        except TypeError:
            return ddgs.text(query, max_results=max_results)

    def _chat_completion(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Call OpenAI chat completion and normalize content to text."""
        request: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if response_format is not None:
            request["response_format"] = response_format

        response = self._get_client().chat.completions.create(**request)
        content = response.choices[0].message.content

        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                text = block.get("text") if isinstance(block, dict) else None
                if not text:
                    text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _parse_json_payload(raw_text: str) -> dict[str, Any]:
        """Parse JSON payload from model output."""
        text = raw_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()
        return json.loads(text)

    @staticmethod
    def _make_source_label(title: str, url: str) -> str:
        """Build a compact human-readable source label."""
        if title:
            return title[:80]
        hostname = urlparse(url).netloc.lower().replace("www.", "")
        return hostname or "source"

    @staticmethod
    def _fallback_summary(
        results: list[dict[str, str]],
        sources: list[dict[str, str]],
    ) -> str:
        """Fallback formatter when model summary has no citations."""
        lines = ["Search findings:"]
        for result, source in zip(results, sources, strict=False):
            snippet = result["snippet"] or result["title"] or source["value"]
            lines.append(f"- {snippet} [{source['label']}]({source['short_url']})")
        return "\n".join(lines)

    @staticmethod
    def _truncate_query(query: str, *, max_length: int) -> str:
        """Truncate long query text for safer search/provider logs."""
        if len(query) <= max_length:
            return query
        truncated = query[:max_length].rsplit(" ", 1)[0].strip()
        if not truncated:
            truncated = query[:max_length].strip()
        return f"{truncated}..."

    def _get_client(self) -> OpenAI:
        """Lazily initialize the OpenAI client."""
        if self.client is None:
            key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is not set")

            base_url = self.base_url or os.getenv("OPENAI_BASE_URL")
            client_kwargs: dict[str, str] = {"api_key": key}
            if base_url:
                client_kwargs["base_url"] = base_url

            self.client = OpenAI(**client_kwargs)
        return self.client


llm_client = LLMClient()
