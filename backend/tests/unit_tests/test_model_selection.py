from agent.configuration import Configuration
from agent.model_selection import (
    carry_reasoning_model,
    select_answer_model,
    select_query_model,
    select_reflection_model,
    select_research_model,
)


def test_select_query_model_prefers_reasoning_model() -> None:
    cfg = Configuration(query_generator_model="gpt-4.1-mini")
    state = {"reasoning_model": "gpt-4.1"}
    assert select_query_model(state, cfg) == "gpt-4.1"


def test_select_research_model_falls_back_to_query_generator_model() -> None:
    cfg = Configuration(query_generator_model="gpt-4.1-mini")
    state = {}
    assert select_research_model(state, cfg) == "gpt-4.1-mini"


def test_select_reflection_model_falls_back_to_reflection_model() -> None:
    cfg = Configuration(reflection_model="gpt-4o-mini")
    state = {}
    assert select_reflection_model(state, cfg) == "gpt-4o-mini"


def test_select_answer_model_falls_back_to_answer_model() -> None:
    cfg = Configuration(answer_model="gpt-4.1")
    state = {}
    assert select_answer_model(state, cfg) == "gpt-4.1"


def test_carry_reasoning_model_only_when_present() -> None:
    assert carry_reasoning_model({"reasoning_model": "gpt-4.1"}) == {
        "reasoning_model": "gpt-4.1"
    }
    assert carry_reasoning_model({}) == {}
