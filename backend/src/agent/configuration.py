"""Agent 运行时配置模型。

本模块负责将以下配置来源合并为统一对象：

1. 代码默认值；
2. 运行时 `RunnableConfig.configurable`；
3. 环境变量（优先级最高）。
"""

import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """研究型 Agent 的统一配置载体。

    Attributes
    ----------
    query_generator_model : str
        用于初始查询词生成的模型名称。
    reflection_model : str
        用于反思阶段（判断信息是否充分、补充后续查询）的模型名称。
    answer_model : str
        用于最终答案合成的模型名称。
    number_of_initial_queries : int
        初始查询数量上限。
    max_research_loops : int
        最大研究循环次数，防止在低质量搜索结果上无限迭代。
    """

    query_generator_model: str = Field(
        default="gpt-4.1-mini",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="gpt-4.1-mini",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="gpt-4.1",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 `RunnableConfig` 与环境变量构建配置实例。

        Parameters
        ----------
        config : RunnableConfig or None, optional
            LangGraph 运行时配置对象。若为空，则只使用环境变量与默认值。

        Returns
        -------
        Configuration
            合并后的最终配置对象。

        Notes
        -----
        优先级遵循：

        1. 环境变量（例如 `QUERY_GENERATOR_MODEL`）；
        2. `config["configurable"]`；
        3. Pydantic 字段默认值。
        """
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
