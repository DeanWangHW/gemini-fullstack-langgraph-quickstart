import argparse
import json
from typing import Any

from langchain_core.messages import HumanMessage

from agent.graph import graph


def _format_update(update: Any, pretty: bool) -> str:
    """Format a graph update payload for terminal output."""
    if pretty:
        return json.dumps(update, ensure_ascii=False, default=str, indent=2)
    return json.dumps(update, ensure_ascii=False, default=str)


def main() -> None:
    """Run the research agent and stream node-by-node updates."""
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
