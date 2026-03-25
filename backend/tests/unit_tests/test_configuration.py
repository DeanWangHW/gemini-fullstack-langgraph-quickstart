from agent.configuration import Configuration


def test_defaults_use_openai_models() -> None:
    cfg = Configuration()
    assert cfg.query_generator_model == "gpt-4.1-mini"
    assert cfg.reflection_model == "gpt-4.1-mini"
    assert cfg.answer_model == "gpt-4.1"


def test_from_runnable_config_reads_configurable_values() -> None:
    cfg = Configuration.from_runnable_config(
        {
            "configurable": {
                "query_generator_model": "gpt-4o-mini",
                "reflection_model": "gpt-4.1",
                "answer_model": "gpt-4.1",
                "number_of_initial_queries": 4,
                "max_research_loops": 5,
            }
        }
    )
    assert cfg.query_generator_model == "gpt-4o-mini"
    assert cfg.reflection_model == "gpt-4.1"
    assert cfg.answer_model == "gpt-4.1"
    assert cfg.number_of_initial_queries == 4
    assert cfg.max_research_loops == 5


def test_from_runnable_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("QUERY_GENERATOR_MODEL", "env-model")
    monkeypatch.setenv("NUMBER_OF_INITIAL_QUERIES", "7")
    cfg = Configuration.from_runnable_config(
        {"configurable": {"query_generator_model": "config-model"}}
    )
    assert cfg.query_generator_model == "env-model"
    assert cfg.number_of_initial_queries == 7
