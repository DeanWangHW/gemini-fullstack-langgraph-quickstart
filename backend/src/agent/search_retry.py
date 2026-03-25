"""DDG 搜索重试策略定义模块。

本模块负责提供与“搜索重试”相关的可配置项，避免将策略硬编码在
`llm_client.py` 中，便于后续单独调整、A/B 实验或按部署环境切换。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchRetryPolicy:
    """搜索重试策略数据结构。

    Attributes
    ----------
    backends : tuple[str | None, ...]
        DDG 查询时尝试的 backend 顺序。
        `None` 表示客户端默认 backend。
    max_error_messages : int
        当所有 backend 都失败时，最多向上游暴露多少条错误摘要。
    """

    backends: tuple[str | None, ...] = (None, "html", "lite", "bing")
    max_error_messages: int = 2


# 默认策略：先默认 backend，再 HTML/Lite/Bing 逐步回退。
DEFAULT_SEARCH_RETRY_POLICY = SearchRetryPolicy()


def get_ddg_retry_backends(policy: SearchRetryPolicy) -> tuple[str | None, ...]:
    """获取 DDG backend 尝试顺序。

    Parameters
    ----------
    policy : SearchRetryPolicy
        当前启用的重试策略对象。

    Returns
    -------
    tuple[str | None, ...]
        backend 顺序元组。
    """
    return policy.backends


def truncate_provider_errors(
    errors: list[str],
    policy: SearchRetryPolicy,
) -> list[str]:
    """按策略截断错误列表长度。

    Parameters
    ----------
    errors : list[str]
        原始错误信息列表。
    policy : SearchRetryPolicy
        当前启用的重试策略对象。

    Returns
    -------
    list[str]
        截断后的错误列表（长度不超过 `policy.max_error_messages`）。
    """
    return errors[: policy.max_error_messages]
