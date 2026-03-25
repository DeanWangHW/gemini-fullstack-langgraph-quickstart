"""结构化输出 Schema 定义模块。

本模块存放所有供 LLM 结构化解析使用的 Pydantic 模型，确保：

1. 节点间数据契约稳定；
2. 提示词与解析逻辑解耦；
3. 单元测试可以直接校验字段语义。
"""

from typing import List

from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    """查询生成节点的结构化返回体。

    Attributes
    ----------
    query : list[str]
        用于网页检索的查询列表。
    rationale : str
        生成这些查询的简要原因说明，便于调试和可解释性分析。
    """

    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    """反思节点的结构化返回体。

    Attributes
    ----------
    is_sufficient : bool
        当前摘要信息是否足够回答用户问题。
    knowledge_gap : str
        若信息不足，说明还缺哪些关键事实。
    follow_up_queries : list[str]
        用于补齐知识缺口的下一轮检索查询。
    """

    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )
