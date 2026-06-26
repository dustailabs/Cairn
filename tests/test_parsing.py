import pytest

from cairn.crew.parsing import (
    CrewOutputError,
    format_retrieved_context,
    parse_critic_output,
    strip_code_fence,
    validate_citations,
)


def test_strip_code_fence_removes_json_fence():
    raw = '```json\n{"answer": "hi"}\n```'
    assert strip_code_fence(raw) == '{"answer": "hi"}'


def test_strip_code_fence_passes_through_unfenced_text():
    raw = '{"answer": "hi"}'
    assert strip_code_fence(raw) == raw


def test_parse_critic_output_happy_path():
    raw = '{"answer": "RRF fuses ranks.", "citations": ["doc::0"], "unresolved": false}'
    result = parse_critic_output(raw)
    assert result.answer == "RRF fuses ranks."
    assert result.citations == ["doc::0"]
    assert result.unresolved is False


def test_parse_critic_output_handles_markdown_fence():
    raw = '```json\n{"answer": "fused via RRF", "citations": []}\n```'
    result = parse_critic_output(raw)
    assert result.answer == "fused via RRF"
    assert result.unresolved is False  # defaults when omitted


def test_parse_critic_output_rejects_invalid_json():
    with pytest.raises(CrewOutputError):
        parse_critic_output("this is not json at all")


def test_parse_critic_output_rejects_missing_answer_field():
    with pytest.raises(CrewOutputError):
        parse_critic_output('{"citations": []}')


def test_parse_critic_output_rejects_non_list_citations():
    with pytest.raises(CrewOutputError):
        parse_critic_output('{"answer": "x", "citations": "doc::0"}')


def test_validate_citations_splits_valid_and_dropped():
    valid, dropped = validate_citations(["a", "b", "z"], {"a", "b"})
    assert valid == ["a", "b"]
    assert dropped == ["z"]


def test_validate_citations_all_dropped_when_none_match():
    valid, dropped = validate_citations(["x", "y"], {"a"})
    assert valid == []
    assert dropped == ["x", "y"]


def test_format_retrieved_context_includes_chunk_id_and_source():
    rendered = format_retrieved_context([("doc::0", "kafka-doc", "Kafka replicates partitions.")])
    assert "[doc::0]" in rendered
    assert "kafka-doc" in rendered
    assert "Kafka replicates partitions." in rendered
