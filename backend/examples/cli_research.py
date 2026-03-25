"""命令行研究脚本（非流式）。

该脚本用于在本地终端快速验证后端图流程是否可用，适合：

1. 验证 `.env` 与模型配置是否生效；
2. 检查图执行结果是否能返回最终答案；
3. 作为 CI 之外的人机快速冒烟入口。
"""

import argparse
from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    """命令行方式运行研究 Agent。

    Parameters
    ----------
    None
        通过 `argparse` 从命令行读取参数。

    Returns
    -------
    None
        函数直接将最终回答打印到标准输出。
    """
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
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
        help="Model for the final answer",
    )
    args = parser.parse_args()

    state = {
        "messages": [HumanMessage(content=args.question)],
        "initial_search_query_count": args.initial_queries,
        "max_research_loops": args.max_loops,
        "reasoning_model": args.reasoning_model,
    }

    result = graph.invoke(state)
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)


if __name__ == "__main__":
    main()
