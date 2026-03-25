from agent.search_retry import (
    DEFAULT_SEARCH_RETRY_POLICY,
    get_ddg_retry_backends,
    truncate_provider_errors,
)


def test_get_ddg_retry_backends_returns_default_order() -> None:
    assert get_ddg_retry_backends(DEFAULT_SEARCH_RETRY_POLICY) == (
        None,
        "html",
        "lite",
        "bing",
    )


def test_truncate_provider_errors_respects_policy_limit() -> None:
    errors = ["e1", "e2", "e3"]
    assert truncate_provider_errors(errors, DEFAULT_SEARCH_RETRY_POLICY) == ["e1", "e2"]
