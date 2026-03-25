from fastapi.staticfiles import StaticFiles
from starlette.routing import Route

from agent.app import create_frontend_router


def test_create_frontend_router_returns_staticfiles_for_valid_build(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
    router = create_frontend_router(str(tmp_path))
    assert isinstance(router, StaticFiles)


def test_create_frontend_router_returns_dummy_route_when_missing_build(tmp_path) -> None:
    missing_dir = tmp_path / "missing"
    router = create_frontend_router(str(missing_dir))
    assert isinstance(router, Route)
