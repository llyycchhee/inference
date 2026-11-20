# CI: linux
import pytest

from ..abstract_tool_parser import request_tools_var
from ..deepseek_v4_tool_parser import DeepseekV4ToolParser


@pytest.fixture
def tools_context():
    token = request_tools_var.set(
        [
            {
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "properties": {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer"},
                        }
                    },
                }
            }
        ]
    )
    yield
    request_tools_var.reset(token)


@pytest.mark.parametrize(
    "opening,closing",
    [("", ""), ("｜DSML｜", ""), ("", "｜DSML｜"), ("｜DSML｜", "｜DSML｜")],
)
@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize("size", [1, 7, 64, 1024])
def test_orphan_and_mixed_tags(tools_context, opening, closing, wrapped, size):
    body = (
        f'<{opening}invoke name="read_file">'
        '<parameter name="path">404</｜DSML｜parameter>'
        '<｜DSML｜parameter name="start_line">160</parameter>'
        f"</{closing}invoke>"
    )
    text = f"<｜DSML｜tool_calls>{body}</tool_calls>" if wrapped else body
    parser = DeepseekV4ToolParser()
    expected = [(None, "read_file", {"path": "404", "start_line": 160})]
    assert parser.extract_tool_calls(text) == expected
    current = ""
    calls = []
    for offset in range(0, len(text), size):
        delta = text[offset : offset + size]
        current += delta
        result = parser.extract_tool_calls_streaming([], current, delta)
        for event in result if isinstance(result, list) else [result] if result else []:
            if event[1]:
                calls.append(event)
    assert calls == expected


@pytest.mark.parametrize(
    "text",
    [
        '<invoke name="unknown"><parameter name="path">x</parameter></invoke>',
        'Example: <invoke name="read_file"><parameter name="path">x</parameter></invoke>',
        '```xml\n<invoke name="read_file"><parameter name="path">x</parameter></invoke>\n```',
        '<invoke name="read_file"><parameter name="path">unfinished',
    ],
)
def test_ambiguous_or_incomplete_xml_stays_text(tools_context, text):
    parser = DeepseekV4ToolParser()
    assert parser.extract_tool_calls(text) == [(text, None, None)]
    current = ""
    content = []
    for delta in text:
        current += delta
        result = parser.extract_tool_calls_streaming([], current, delta)
        for event in result if isinstance(result, list) else [result] if result else []:
            assert event[1] is None
            content.append(event[0] or "")
    content.append(parser.flush_streaming_content(current) or "")
    assert "".join(content) == text


def test_plain_orphan_requires_request_tools():
    token = request_tools_var.set(None)
    try:
        text = '<invoke name="read_file"></invoke>'
        assert DeepseekV4ToolParser().extract_tool_calls(text) == [(text, None, None)]
    finally:
        request_tools_var.reset(token)


def test_duplicate_closers_and_parallel_calls(tools_context):
    parser = DeepseekV4ToolParser()
    call = '<invoke name="read_file"><parameter name="path">x</parameter></invoke>'
    text = call + "\n</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
    assert parser.extract_tool_calls(text) == [(None, "read_file", {"path": "x"})]
    assert (
        parser.extract_tool_calls(call + call)
        == [(None, "read_file", {"path": "x"})] * 2
    )


@pytest.mark.parametrize(
    "text",
    [
        '<invoke name="read_',
        '<｜DSML｜invoke name="read_file">',
        '<invoke name="read_file"><parameter name="path">x</parameter></｜DSML｜tool_calls>',
    ],
)
def test_incomplete_candidate_rolls_back(tools_context, text):
    test_ambiguous_or_incomplete_xml_stays_text(tools_context, text)


def test_recovered_call_preserves_suffix_and_checks_each_name(tools_context):
    parser = DeepseekV4ToolParser()
    good = '<invoke name="read_file"><parameter name="path">x</parameter></invoke>'
    bad = '<invoke name="unknown"></invoke>'
    text = good + "\n" + bad + " suffix"
    expected = [
        (None, "read_file", {"path": "x"}),
        ("\n" + bad + " suffix", None, None),
    ]
    assert parser.extract_tool_calls(text) == expected
    current = ""
    events = []
    for delta in text:
        current += delta
        result = parser.extract_tool_calls_streaming([], current, delta)
        events.extend(
            result if isinstance(result, list) else [result] if result else []
        )
    assert [e for e in events if e[1]] == expected[:1]
    assert "".join(e[0] or "" for e in events) == "\n" + bad + " suffix"


def test_invoke_markup_in_parameter_value_is_opaque(tools_context):
    parser = DeepseekV4ToolParser()
    value = 'literal <invoke name="unknown"></invoke> code'
    text = (
        '<invoke name="read_file"><parameter name="path">'
        + value
        + "</parameter></invoke>"
    )
    assert parser.extract_tool_calls(text) == [(None, "read_file", {"path": value})]


@pytest.mark.parametrize("prefix", ["", "｜DSML｜"])
def test_no_tools_never_recovers_orphan(prefix):
    token = request_tools_var.set(None)
    try:
        text = f'<{prefix}invoke name="read_file"></{prefix}invoke>'
        assert DeepseekV4ToolParser().extract_tool_calls(text) == [(text, None, None)]
    finally:
        request_tools_var.reset(token)


def test_orphan_recovery_is_request_local():
    parser = DeepseekV4ToolParser()
    texts = [
        '<invoke name="read_file"><parameter name="path">a</parameter></invoke>',
        '<invoke name="read_file"><parameter name="path">b</parameter></invoke>',
    ]
    histories = ["", ""]
    events = [[], []]
    for offset in range(max(map(len, texts))):
        for index, text in enumerate(texts):
            delta = text[offset : offset + 1]
            histories[index] += delta
            token = request_tools_var.set(
                [{"function": {"name": "read_file"}}] if index == 0 else []
            )
            try:
                result = parser.extract_tool_calls_streaming(
                    [], histories[index], delta
                )
            finally:
                request_tools_var.reset(token)
            events[index].extend(
                result if isinstance(result, list) else [result] if result else []
            )
    assert [e for e in events[0] if e[1]] == [(None, "read_file", {"path": "a"})]
    assert not any(e[1] for e in events[1])
    assert "".join(e[0] or "" for e in events[1]) == texts[1]


@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize("literal", ["</invoke>", "</tool_calls>"])
@pytest.mark.parametrize("size", [1, 7, 64, 1024])
def test_literal_invoke_closer_in_parameter(tools_context, wrapped, literal, size):
    parser = DeepseekV4ToolParser()
    value = "literal " + literal + " end"
    body = (
        '<｜DSML｜invoke name="read_file">'
        '<｜DSML｜parameter name="path" string="true">'
        + value
        + "</｜DSML｜parameter></｜DSML｜invoke>"
    )
    text = "<｜DSML｜tool_calls>" + body + "</｜DSML｜tool_calls>" if wrapped else body
    expected = [(None, "read_file", {"path": value})]
    assert parser.extract_tool_calls(text) == expected
    assert _review_stream(parser, text, size)[0] == expected


def _review_stream(parser, text, size):
    current, events = "", []
    for offset in range(0, len(text), size):
        delta = text[offset : offset + size]
        current += delta
        result = parser.extract_tool_calls_streaming([], current, delta)
        events.extend(
            result if isinstance(result, list) else [result] if result else []
        )
    flush = parser.flush_streaming_content(current)
    calls = [event for event in events if event[1]]
    content = "".join(event[0] or "" for event in events) + (flush or "")
    return calls, content


@pytest.mark.parametrize("space", [" ", "  ", "\n", "\t", "\r\n  "])
@pytest.mark.parametrize("prefix", ["", "｜DSML｜"])
@pytest.mark.parametrize("size", [1, 7, 1024])
def test_orphan_header_whitespace_does_not_leak(tools_context, space, prefix, size):
    text = (
        f'<{prefix}invoke{space}name="read_file">'
        '<parameter name="path">x</parameter></invoke>'
    )
    calls, content = _review_stream(DeepseekV4ToolParser(), text, size)
    assert calls == [(None, "read_file", {"path": "x"})]
    assert content == ""


@pytest.mark.parametrize(
    "suffix", ["suffix </par", "suffix </｜DSML｜inv", "suffix <", "suffix <invoke\nna"]
)
@pytest.mark.parametrize("size", [1, 7, 1024])
def test_stream_flush_preserves_partial_suffix(tools_context, suffix, size):
    text = '<invoke name="read_file"></invoke>' + suffix
    parser = DeepseekV4ToolParser()
    calls, content = _review_stream(parser, text, size)
    assert calls == [(None, "read_file", {})]
    assert content == suffix


@pytest.mark.parametrize(
    "text", ["<invoke\nna", '<invoke\tname="read_file', "<invoke  "]
)
def test_stream_flush_preserves_incomplete_whitespace_header(tools_context, text):
    calls, content = _review_stream(DeepseekV4ToolParser(), text, 1)
    assert calls == []
    assert content == text
