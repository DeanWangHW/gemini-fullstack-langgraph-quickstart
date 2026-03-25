# mypy: disable - error - code = "no-untyped-def,misc"
"""FastAPI 入口与前端静态资源挂载模块。

本模块只负责 HTTP 层启动逻辑，不承担 Agent 推理逻辑。核心职责是：

1. 创建 FastAPI 应用实例；
2. 将前端构建产物（Vite `dist`）以静态文件形式挂载到 `/app`；
3. 在前端未构建时，返回可读的 503 文本，降低排障成本。
"""

import pathlib
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

# 定义全局 FastAPI 应用实例，供 langgraph.json 的 http 入口直接引用。
app = FastAPI()


def create_frontend_router(build_dir="../frontend/dist"):
    """创建并返回用于服务前端页面的路由对象。

    Parameters
    ----------
    build_dir : str, optional
        前端构建目录（相对本文件路径）。默认值为 `../frontend/dist`。

    Returns
    -------
    fastapi.staticfiles.StaticFiles or starlette.routing.Route
        当构建目录可用时返回 `StaticFiles`，用于提供 SPA 静态资源；
        当构建目录不可用时返回兜底 `Route`，并在访问时返回 503 提示信息。

    Notes
    -----
    - 该函数在应用启动时执行，避免首次请求时才发现前端资源缺失。
    - 返回类型为二选一，调用方可统一以 Starlette 路由接口消费。
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # 构建结果不存在时，返回明确错误信息，便于开发环境快速定位问题。
        from starlette.routing import Route

        async def dummy_frontend(request):
            """前端缺失时的简易降级处理。

            Parameters
            ----------
            request : starlette.requests.Request
                当前 HTTP 请求对象（此处不使用，但保留签名以匹配路由协议）。

            Returns
            -------
            fastapi.Response
                状态码 503 的纯文本响应，提示用户先构建前端。
            """
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)


# 将前端挂载到 /app，避免与 LangGraph API 路径冲突。
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
