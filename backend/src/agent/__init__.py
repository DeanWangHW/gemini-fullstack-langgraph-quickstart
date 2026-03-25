"""Agent 包导出入口。

该模块对外仅暴露编译后的 `graph` 对象，供 CLI、HTTP 服务与测试统一引用。
"""

from agent.graph import graph

__all__ = ["graph"]
