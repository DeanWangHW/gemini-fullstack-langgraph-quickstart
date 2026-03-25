"""与对话上下文和引用处理相关的通用工具函数。"""

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """根据消息列表构造研究主题文本。

    Parameters
    ----------
    messages : list[AnyMessage]
        输入消息列表。通常至少包含一条用户消息。

    Returns
    -------
    str
        若仅一条消息，则直接返回该消息内容；
        若有多轮对话，则按 ``User/Assistant`` 前缀拼接为上下文文本。

    Notes
    -----
    该函数用于把多轮对话压平成可直接喂给提示词模板的文本片段，
    以保留上下文语义并降低跨轮信息丢失风险。
    """
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """将原始长 URL 映射为短链接占位符。

    Parameters
    ----------
    urls_to_resolve : list[Any]
        含有 ``site.web.uri`` 字段的对象列表。
    id : int
        当前查询批次 ID，用于短链接命名空间隔离。

    Returns
    -------
    dict[str, str]
        原始 URL 到短链接的映射字典。同一原始 URL 始终映射为同一短链接。
    """
    prefix = "https://vertexaisearch.cloud.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]

    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map


def insert_citation_markers(text: str, citations_list: List[Dict[str, Any]]) -> str:
    """在正文中插入引用标记（Markdown 链接）。

    Parameters
    ----------
    text : str
        原始文本。
    citations_list : list[dict[str, Any]]
        引用区间列表。每个元素至少包含：

        - ``start_index``：引用文本起始位置；
        - ``end_index``：引用文本结束位置；
        - ``segments``：引用来源段列表（含 ``label`` 和 ``short_url``）。

    Returns
    -------
    str
        插入引用标记后的文本。

    Notes
    -----
    为避免插入操作破坏后续区间索引，函数按 ``end_index`` 倒序处理。
    """
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text


def get_citations(response: Any, resolved_urls_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """从模型响应中提取并格式化引用信息。

    Parameters
    ----------
    response : Any
        模型返回对象。历史上该结构来自 Gemini grounding 输出，
        需要包含 ``candidates[0].grounding_metadata``。
    resolved_urls_map : dict[str, str]
        原始 URL 到短链接的映射。

    Returns
    -------
    list[dict[str, Any]]
        标准化后的引用列表。每条引用包含：

        - ``start_index``：引用起始字符位置；
        - ``end_index``：引用结束字符位置；
        - ``segments``：来源片段列表（每项含 label/short_url/value）。

    Notes
    -----
    该函数保留是为了兼容旧响应结构。当前 OpenAI 路径未直接使用该函数，
    但对历史分支或迁移脚本仍有参考价值。
    """
    citations: List[Dict[str, Any]] = []

    if not response or not response.candidates:
        return citations

    candidate = response.candidates[0]
    if (
        not hasattr(candidate, "grounding_metadata")
        or not candidate.grounding_metadata
        or not hasattr(candidate.grounding_metadata, "grounding_supports")
    ):
        return citations

    for support in candidate.grounding_metadata.grounding_supports:
        citation: Dict[str, Any] = {}

        if not hasattr(support, "segment") or support.segment is None:
            continue

        start_index = (
            support.segment.start_index
            if support.segment.start_index is not None
            else 0
        )

        if support.segment.end_index is None:
            continue

        citation["start_index"] = start_index
        citation["end_index"] = support.segment.end_index

        citation["segments"] = []
        if (
            hasattr(support, "grounding_chunk_indices")
            and support.grounding_chunk_indices
        ):
            for ind in support.grounding_chunk_indices:
                try:
                    chunk = candidate.grounding_metadata.grounding_chunks[ind]
                    resolved_url = resolved_urls_map.get(chunk.web.uri, None)
                    citation["segments"].append(
                        {
                            "label": chunk.web.title.split(".")[:-1][0],
                            "short_url": resolved_url,
                            "value": chunk.web.uri,
                        }
                    )
                except (IndexError, AttributeError, NameError):
                    # 兼容脏数据：单个 chunk 解析失败时跳过，不中断整体流程。
                    pass
        citations.append(citation)
    return citations
