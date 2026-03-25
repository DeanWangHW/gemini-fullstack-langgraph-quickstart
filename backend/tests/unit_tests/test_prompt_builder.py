from agent.prompt_builder import (
    build_answer_prompt,
    build_query_prompt,
    build_reflection_prompt,
)


def test_build_query_prompt_renders_topic_count_and_date() -> None:
    prompt = build_query_prompt(
        research_topic="battery prices",
        number_queries=2,
        current_date="March 25, 2026",
    )
    assert "battery prices" in prompt
    assert "March 25, 2026" in prompt
    assert "2" in prompt


def test_build_reflection_prompt_renders_summaries() -> None:
    prompt = build_reflection_prompt(
        research_topic="wind trends",
        summaries=["s1", "s2"],
        current_date="March 25, 2026",
    )
    assert "wind trends" in prompt
    assert "s1" in prompt
    assert "s2" in prompt


def test_build_answer_prompt_renders_summaries() -> None:
    prompt = build_answer_prompt(
        research_topic="clean energy investment",
        summaries=["summary block"],
        current_date="March 25, 2026",
    )
    assert "clean energy investment" in prompt
    assert "summary block" in prompt
