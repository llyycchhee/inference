# CI: linux
import pytest

from ..abstract_tool_parser import request_tool_state_var, request_tools_var
from ..deepseek_v4_tool_parser import DeepseekV4ToolParser

TOOLS = [
    {
        "function": {
            "name": "write",
            "parameters": {"properties": {"code": {"type": "string"}}},
        }
    }
]


def stream(text, size):
    parser = DeepseekV4ToolParser()
    state = {}
    tool_token = request_tools_var.set(TOOLS)
    state_token = request_tool_state_var.set(state)
    current, events = "", []
    try:
        for offset in range(0, len(text), size):
            delta = text[offset : offset + size]
            previous = [current]
            current += delta
            result = parser.extract_tool_calls_streaming(previous, current, delta)
            events.extend(
                result if isinstance(result, list) else [result] if result else []
            )
        suffix = parser.flush_streaming_content(current)
        assert parser.flush_streaming_content(current) is None
        if suffix:
            events.append((suffix, None, None))
        assert state["deepseek_v4_stream"].seen == len(text)
        return [event for event in events if event[1]], "".join(
            event[0] or "" for event in events
        )
    finally:
        request_tool_state_var.reset(state_token)
        request_tools_var.reset(tool_token)


@pytest.mark.parametrize("size", [1, 7, 64, 1024])
@pytest.mark.parametrize("prefix", ["", "｜DSML｜"])
@pytest.mark.parametrize("space", [" ", "\n", "\t", "  "])
def test_incremental_orphan(size, prefix, space):
    text = (
        "before\n" + f'<{prefix}invoke{space}name="write">'
        '<parameter name="code">literal </invoke> end</｜DSML｜parameter>'
        "</invoke>suffix </par"
    )
    calls, content = stream(text, size)
    assert calls == [(None, "write", {"code": "literal </invoke> end"})]
    assert content == "before\nsuffix </par"


@pytest.mark.parametrize("size", [1, 7, 64, 1024])
@pytest.mark.parametrize("close", ["</parameter>", "</｜DSML｜parameter>"])
def test_incremental_wrapped_mixed_value(size, close):
    value = "code </invoke> </tool_calls> tail"
    text = (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="write">'
        '<｜DSML｜parameter name="code" string="true">'
        + value
        + close
        + "</invoke></tool_calls>"
    )
    assert stream(text, size) == ([(None, "write", {"code": value})], "")


@pytest.mark.parametrize(
    "text",
    [
        '<invoke name="unknown"></invoke>',
        'Example: <invoke name="write"></invoke>',
        '```xml\n<invoke name="write"></invoke>\n```',
        "<invoke\nna",
        '<invoke name="write"><parameter name="code">unfinished',
        '<invoke name="write"><parameter name="code">x</parameter></tool_calls>',
        "plain text with < and </par",
    ],
)
@pytest.mark.parametrize("size", [1, 7, 1024])
def test_incremental_rollback_and_plain_text(text, size):
    assert stream(text, size) == ([], text)


def test_incremental_avoids_stateless_rescan(monkeypatch):
    def fail(*args):
        raise AssertionError("stateless full-output scan used")

    monkeypatch.setattr(DeepseekV4ToolParser, "_invoke_frames", fail)
    text = (
        '<invoke name="write"><parameter name="code">'
        + "x" * 20000
        + "</parameter></invoke>"
    )
    assert stream(text, 1) == ([(None, "write", {"code": "x" * 20000})], "")


def test_incremental_request_isolation():
    parser = DeepseekV4ToolParser()
    texts = ['<invoke name="write"></invoke>', '<invoke name="unknown"></invoke>']
    contexts, currents, events = [{}, {}], ["", ""], [[], []]
    tools_token = request_tools_var.set(TOOLS)
    try:
        for offset in range(max(map(len, texts))):
            for i in range(2):
                delta = texts[i][offset : offset + 1]
                if not delta:
                    continue
                old = currents[i]
                currents[i] += delta
                token = request_tool_state_var.set(contexts[i])
                try:
                    result = parser.extract_tool_calls_streaming(
                        [old], currents[i], delta
                    )
                    events[i].extend(
                        result
                        if isinstance(result, list)
                        else [result] if result else []
                    )
                finally:
                    request_tool_state_var.reset(token)
        assert [e for e in events[0] if e[1]] == [(None, "write", {})]
        assert not any(e[1] for e in events[1])
        assert "".join(e[0] or "" for e in events[1]) == texts[1]
        assert (
            contexts[0]["deepseek_v4_stream"] is not contexts[1]["deepseek_v4_stream"]
        )
    finally:
        request_tools_var.reset(tools_token)


@pytest.mark.parametrize("close", ["</parameter>", "</｜DSML｜parameter>"])
def test_chat_path_keeps_early_arguments_with_mixed_closer(close):
    import json

    from ...utils import ChatModelMixin
    from .test_argument_stream import deltas, feed

    model = ChatModelMixin()
    model.tool_parser = DeepseekV4ToolParser()
    history, state, calls = [""], {}, []
    opening = (
        '<｜DSML｜tool_calls><｜DSML｜invoke name="write">'
        '<｜DSML｜parameter name="code" string="true">'
    )
    prefix = "first </invoke> part"
    for delta in [opening, prefix]:
        calls.extend(deltas(feed(model, delta, history, state)))
    partial = "".join(c["function"].get("arguments", "") for c in calls)
    assert "first" in partial
    with pytest.raises(json.JSONDecodeError):
        json.loads(partial)
    for delta in [close, "</invoke>", "</tool_calls>"]:
        calls.extend(deltas(feed(model, delta, history, state)))
    assert json.loads("".join(c["function"].get("arguments", "") for c in calls)) == {
        "code": prefix
    }
    assert sum("name" in c["function"] for c in calls) == 1
    assert (
        feed(model, "", history, state, "stop")["choices"][0]["finish_reason"]
        == "tool_calls"
    )
    assert request_tool_state_var.get() is None


@pytest.mark.parametrize("size", [1, 7, 64, 1024])
@pytest.mark.parametrize(
    "marker",
    [
        "</parameter>",
        "</invoke>",
        "</tool_calls>",
        "</｜DSML｜parameter>",
        "</｜DSML｜invoke>",
        "</｜DSML｜tool_calls>",
    ],
)
@pytest.mark.parametrize("prefix", [" XML: ", "\nplain text ", "<literal>"])
def test_prose_ends_structural_tail_absorption(size, marker, prefix):
    text = '<invoke name="write"></invoke>' + prefix + marker + " tail"
    assert stream(text, size) == ([(None, "write", {})], prefix + marker + " tail")


@pytest.mark.parametrize("size", [1, 7, 64, 1024])
def test_adjacent_closers_absorbed_before_prose(size):
    text = (
        '<invoke name="write"></invoke> \n</｜DSML｜parameter>'
        "</invoke></｜DSML｜tool_calls> XML: </parameter>"
    )
    calls, content = stream(text, size)
    assert calls == [(None, "write", {})]
    assert content.strip() == "XML: </parameter>"
