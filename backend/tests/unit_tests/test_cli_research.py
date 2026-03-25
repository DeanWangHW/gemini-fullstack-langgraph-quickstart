"""`examples/cli_research.py` 脚本单元测试。

该文件主要验证 CLI 参数是否被正确转换为图状态，并确保脚本会输出
最终回答文本，避免命令行入口在重构中失效。
"""

import importlib.util
import sys
from pathlib import Path

from langchain_core.messages import AIMessage


def _load_cli_module():
    """按路径动态加载 CLI 脚本模块。

    Returns
    -------
    module
        已加载的 `cli_research.py` 模块对象。
    """
    backend_root = Path(__file__).resolve().parents[2]
    script = backend_root / "examples" / "cli_research.py"
    spec = importlib.util.spec_from_file_location("cli_research", script)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_main_prints_final_answer(monkeypatch, capsys) -> None:
    """验证 CLI 主函数会打印最终回答。

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        用于替换 `graph` 与命令行参数。
    capsys : pytest.CaptureFixture[str]
        用于捕获标准输出。

    Returns
    -------
    None
        断言通过即表示行为符合预期。
    """
    module = _load_cli_module()

    class DummyGraph:
        """用于替换真实图对象的最小桩实现。"""

        def invoke(self, state):
            """校验输入状态并返回固定答案。"""
            assert state["initial_search_query_count"] == 1
            assert state["max_research_loops"] == 1
            assert state["reasoning_model"] == "gpt-4.1-mini"
            return {"messages": [AIMessage(content="final answer")]}

    monkeypatch.setattr(module, "graph", DummyGraph())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_research.py",
            "What is LangGraph?",
            "--initial-queries",
            "1",
            "--max-loops",
            "1",
            "--reasoning-model",
            "gpt-4.1-mini",
        ],
    )

    module.main()
    output = capsys.readouterr().out
    assert "final answer" in output
