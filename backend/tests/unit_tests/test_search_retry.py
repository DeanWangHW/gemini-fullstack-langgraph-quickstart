"""`agent.search_retry` 模块单元测试。

本文件确保检索重试策略的两个核心行为稳定：

1. backend 尝试顺序符合默认策略；
2. provider 错误信息会按策略上限截断。
"""

from agent.search_retry import (
    DEFAULT_SEARCH_RETRY_POLICY,
    get_ddg_retry_backends,
    truncate_provider_errors,
)


def test_get_ddg_retry_backends_returns_default_order() -> None:
    """验证默认 backend 顺序未被意外修改。"""
    assert get_ddg_retry_backends(DEFAULT_SEARCH_RETRY_POLICY) == (
        None,
        "html",
        "lite",
        "bing",
    )


def test_truncate_provider_errors_respects_policy_limit() -> None:
    """验证错误摘要数量按策略上限裁剪。"""
    errors = ["e1", "e2", "e3"]
    assert truncate_provider_errors(errors, DEFAULT_SEARCH_RETRY_POLICY) == ["e1", "e2"]
