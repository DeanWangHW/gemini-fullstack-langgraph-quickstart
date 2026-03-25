import importlib.util
import sys
from pathlib import Path

from langchain_core.messages import AIMessage


def _load_cli_module():
    backend_root = Path(__file__).resolve().parents[2]
    script = backend_root / "examples" / "cli_research.py"
    spec = importlib.util.spec_from_file_location("cli_research", script)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_main_prints_final_answer(monkeypatch, capsys) -> None:
    module = _load_cli_module()

    class DummyGraph:
        def invoke(self, state):
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
