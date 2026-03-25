"""命令行研究脚本（流式节点更新版）。

与 `cli_research.py` 相比，本脚本会逐节点输出 LangGraph 运行中的状态增量，
用于调试以下问题：

1. 查询生成是否符合预期；
2. 网页检索阶段是否拿到来源与摘要；
3. 反思路由是否会触发后续查询。
"""

import argparse
import json
from typing import Any

from langchain_core.messages import HumanMessage

from agent.graph import graph


def _format_update(update: Any, pretty: bool) -> str:
    """格式化图节点更新信息。

    Parameters
    ----------
    update : Any
        节点更新对象。
    pretty : bool
        是否使用缩进格式输出 JSON。

    Returns
    -------
    str
        格式化后的字符串。
    """
    if pretty:
        return json.dumps(update, ensure_ascii=False, default=str, indent=2)
    return json.dumps(update, ensure_ascii=False, default=str)


def main() -> None:
    """以流式方式运行研究 Agent 并打印节点级更新。

    Parameters
    ----------
    None
        参数通过命令行传入。

    Returns
    -------
    None
        过程输出直接写入标准输出。
    """
    parser = argparse.ArgumentParser(
        description="Run the LangGraph research agent with streaming node updates"
    )
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=3,
        help="Number of initial search queries",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=2,
        help="Maximum number of research loops",
    )
    parser.add_argument(
        "--reasoning-model",
        default="gpt-4.1",
        help="Model used across the research pipeline",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON updates",
    )
    args = parser.parse_args()

    state = {
        "messages": [HumanMessage(content=args.question)],
        "initial_search_query_count": args.initial_queries,
        "max_research_loops": args.max_loops,
        "reasoning_model": args.reasoning_model,
    }

    print("Streaming LangGraph updates...\n")
    final_answer = None

    for chunk in graph.stream(state, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue

        for node_name, update in chunk.items():
            print(f"[node] {node_name}")
            print(f"[update] {_format_update(update, pretty=args.pretty)}\n")

            if isinstance(update, dict) and update.get("messages"):
                last_message = update["messages"][-1]
                content = getattr(last_message, "content", None)
                if isinstance(content, str):
                    final_answer = content

    if final_answer:
        print("=== Final Answer ===")
        print(final_answer)


if __name__ == "__main__":
    main()
