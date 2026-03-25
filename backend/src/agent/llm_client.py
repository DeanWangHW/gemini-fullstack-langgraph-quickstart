"""OpenAI 调用与网页检索摘要客户端。

该模块是后端“模型调用 + 检索摘要”能力的统一入口，负责：

1. 文本生成与结构化生成；
2. DDG 检索、结果归一化与来源结构化；
3. 查询预处理、错误聚合与摘要兜底。
"""

import json
import os
import re
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from agent.search_retry import (
    DEFAULT_SEARCH_RETRY_POLICY,
    get_ddg_retry_backends,
    truncate_provider_errors,
)

T = TypeVar("T", bound=BaseModel)

# Explicitly load backend/.env so variable resolution does not depend on cwd.
BACKEND_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=BACKEND_ENV_PATH)


class LLMClient:
    """封装 OpenAI 调用与网页检索工具链。

    Parameters
    ----------
    api_key : str or None, optional
        显式传入的 OpenAI API Key。为空时从环境变量 `OPENAI_API_KEY` 读取。
    base_url : str or None, optional
        OpenAI 兼容网关地址。为空时从环境变量 `OPENAI_BASE_URL` 读取。

    Notes
    -----
    客户端对象采用延迟初始化策略，首次调用 `_get_client` 时才真正创建 SDK 实例。
    """

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
        """生成普通文本回答。

        Parameters
        ----------
        prompt : str
            输入提示词。
        model : str
            目标模型名称。
        temperature : float, optional
            采样温度，默认 `0.0`。

        Returns
        -------
        str
            模型返回的纯文本内容。
        """
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
        """生成结构化 JSON 并校验为 Pydantic 对象。

        Parameters
        ----------
        prompt : str
            输入提示词。
        schema : type[T]
            目标 Pydantic 模型类型。
        model : str
            目标模型名称。
        temperature : float, optional
            采样温度，默认 `0.0`。

        Returns
        -------
        T
            通过 `schema.model_validate` 校验后的结构化对象。
        """
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
        """执行网页检索并生成带引用占位符的摘要。

        Parameters
        ----------
        query : str
            原始检索查询。
        query_id : int
            当前查询任务 ID，用于生成短链接占位符。
        model : str
            用于摘要生成的模型名称。
        max_results : int, optional
            最多采用的去重结果数，默认 5。

        Returns
        -------
        tuple[str, list[dict[str, str]]]
            二元组 `(summary_text, sources)`：

            - `summary_text`：摘要文本（含短链接引用）；
            - `sources`：来源结构列表（label/short_url/value）。

        Raises
        ------
        ImportError
            当运行环境缺少 `ddgs` 依赖时抛出。

        Notes
        -----
        当检索为空时，函数返回包含错误上下文的可读文本，而不是抛异常，
        以保证图流程可继续进入反思阶段。
        """
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
                summarized_errors = truncate_provider_errors(
                    search_errors,
                    DEFAULT_SEARCH_RETRY_POLICY,
                )
                error_context = " Search provider errors: " + " | ".join(
                    summarized_errors
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
        """在发送到搜索引擎前清洗查询文本。

        Parameters
        ----------
        query : str
            原始查询文本（可能来自 LLM）。

        Returns
        -------
        str
            清洗后的查询文本。会移除“please cite”类非检索关键词，并在过长时压缩。
        """
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
        """将自然语言长问句压缩为关键词查询。

        Parameters
        ----------
        query : str
            原始查询文本。
        max_terms : int, optional
            保留关键词上限，默认 18。

        Returns
        -------
        str
            适合搜索引擎的紧凑关键词字符串。
        """
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
        """执行 DDG 多 backend 检索并聚合结果。

        Parameters
        ----------
        DDGS : Any
            `ddgs.DDGS` 类对象（通过外部注入，便于测试替身）。
        query : str
            已清洗后的查询文本。
        max_results : int
            单 backend 最大返回结果数。

        Returns
        -------
        tuple[list[dict[str, str]], list[str]]
            二元组 `(rows, errors)`：

            - `rows`：归一化后的检索结果；
            - `errors`：各 backend 错误摘要。
        """
        backends = get_ddg_retry_backends(DEFAULT_SEARCH_RETRY_POLICY)
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
        """把 DDG 返回结构归一化为统一记录格式。

        Parameters
        ----------
        items : Any
            DDG 原始返回对象（可能是列表、字典或迭代器）。

        Returns
        -------
        list[dict[str, str]]
            统一后的记录列表，字段为 `title/url/snippet`。
        """
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
        """调用 DDGS 文本搜索，并兼容旧签名。

        Parameters
        ----------
        ddgs : Any
            DDGS 客户端实例。
        query : str
            查询文本。
        max_results : int
            最大结果数。
        backend : str or None
            指定 backend。`None` 表示默认 backend。

        Returns
        -------
        Any
            DDG 原始检索返回对象。
        """
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
        """执行 Chat Completions 调用并统一返回文本。

        Parameters
        ----------
        prompt : str
            输入提示词。
        model : str
            模型名称。
        temperature : float
            采样温度。
        response_format : dict[str, Any] or None, optional
            结构化输出配置（例如 JSON 模式）。

        Returns
        -------
        str
            归一化后的文本内容。

        Notes
        -----
        OpenAI SDK 可能返回字符串、列表块或空值；该函数统一处理为字符串。
        """
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
        """从模型输出中提取 JSON 负载。

        Parameters
        ----------
        raw_text : str
            模型原始文本输出（可能包含 Markdown 代码块包裹）。

        Returns
        -------
        dict[str, Any]
            解析后的 JSON 字典。
        """
        text = raw_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if fenced_match:
            text = fenced_match.group(1).strip()
        return json.loads(text)

    @staticmethod
    def _make_source_label(title: str, url: str) -> str:
        """构建来源展示标签。

        Parameters
        ----------
        title : str
            原始标题。
        url : str
            原始链接。

        Returns
        -------
        str
            若标题存在，返回截断标题；否则返回域名。
        """
        if title:
            return title[:80]
        hostname = urlparse(url).netloc.lower().replace("www.", "")
        return hostname or "source"

    @staticmethod
    def _fallback_summary(
        results: list[dict[str, str]],
        sources: list[dict[str, str]],
    ) -> str:
        """当模型摘要缺少引用时生成保底摘要。

        Parameters
        ----------
        results : list[dict[str, str]]
            检索结果列表。
        sources : list[dict[str, str]]
            对应来源列表。

        Returns
        -------
        str
            带来源链接的基础摘要文本。
        """
        lines = ["Search findings:"]
        for result, source in zip(results, sources, strict=False):
            snippet = result["snippet"] or result["title"] or source["value"]
            lines.append(f"- {snippet} [{source['label']}]({source['short_url']})")
        return "\n".join(lines)

    @staticmethod
    def _truncate_query(query: str, *, max_length: int) -> str:
        """截断过长查询文本，避免日志或 provider 参数异常。

        Parameters
        ----------
        query : str
            原始查询文本。
        max_length : int
            最大允许长度。

        Returns
        -------
        str
            截断后的查询文本。
        """
        if len(query) <= max_length:
            return query
        truncated = query[:max_length].rsplit(" ", 1)[0].strip()
        if not truncated:
            truncated = query[:max_length].strip()
        return f"{truncated}..."

    def _get_client(self) -> OpenAI:
        """获取（或初始化）OpenAI 客户端实例。

        Returns
        -------
        openai.OpenAI
            可复用的 SDK 客户端。

        Raises
        ------
        ValueError
            当未提供 `OPENAI_API_KEY` 时抛出。
        """
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
