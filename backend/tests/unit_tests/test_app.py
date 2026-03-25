"""`agent.app` 模块单元测试。

本文件验证前端静态资源挂载路由在“构建目录存在/不存在”两种情况下的行为，
确保后端启动时能够提供可预期的路由对象类型。
"""

from fastapi.staticfiles import StaticFiles
from starlette.routing import Route

from agent.app import create_frontend_router


def test_create_frontend_router_returns_staticfiles_for_valid_build(tmp_path) -> None:
    """验证构建目录存在时返回 `StaticFiles`。

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest 临时目录夹具。

    Returns
    -------
    None
        断言通过即表示行为符合预期。
    """
    (tmp_path / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
    router = create_frontend_router(str(tmp_path))
    assert isinstance(router, StaticFiles)


def test_create_frontend_router_returns_dummy_route_when_missing_build(tmp_path) -> None:
    """验证构建目录缺失时返回降级 `Route`。

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest 临时目录夹具。

    Returns
    -------
    None
        断言通过即表示行为符合预期。
    """
    missing_dir = tmp_path / "missing"
    router = create_frontend_router(str(missing_dir))
    assert isinstance(router, Route)
