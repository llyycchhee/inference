import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from . import register_tool_parser
from .abstract_tool_parser import ToolParser, request_tools_var
from .argument_stream import ArgumentStart, dsml_argument_start

logger = logging.getLogger(__name__)

ToolCall = Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]


@register_tool_parser("deepseek-v4")
class DeepseekV4ToolParser(ToolParser):
    """
    Tool parser implementation for DeepSeek V4 model.

    Handles two formats used by DeepSeek V4 for tool calls:

    DSML format:
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    Plain format:
    <tool_calls>
    <invoke name="get_weather">
    <parameter name="location" string="true">杭州</parameter>
    <parameter name="date" string="true">2024-01-16</parameter>
    </invoke>
    </tool_calls>
    """

    # Prefix used in DSML-flavoured tags
    _DSML = "<｜DSML｜"

    def __init__(self):
        super().__init__()

        self.tool_call_start_tokens: List[str] = [
            f"{self._DSML}tool_calls>",
            "<tool_calls>",
        ]

        # All patterns are shared and immutable. Per-request recovery is
        # reconstructed from that request's accumulated text.
        self._wrapper_open, self._wrapper_close, self._parameter_regex = (
            self._build_regexes()
        )

    def _build_regexes(self) -> Tuple[re.Pattern, re.Pattern, re.Pattern]:
        """Compile structural patterns accepting plain and DSML spellings."""
        # Each structural tag may independently use the plain or DSML spelling.
        # Do not normalize the output: argument values may contain source code.
        prefix = r"(?:｜DSML｜)?"
        wrapper_open = re.compile(rf"<{prefix}tool_calls>")
        wrapper_close = re.compile(rf"</{prefix}tool_calls>")
        parameter_regex = re.compile(
            rf'<{prefix}parameter\s+name="([^"]+)"'
            rf'(?:\s+string="(true|false)")?\s*>(.*?)</{prefix}parameter>',
            re.DOTALL,
        )
        return wrapper_open, wrapper_close, parameter_regex

    def _parse_invoke_params(
        self,
        invoke_str: str,
        parameter_regex: re.Pattern,
        function_name: Optional[str] = None,
    ) -> dict:
        """Parse parameter name-value pairs from an invoke block."""
        param_dict = {}
        for param_name, is_string, param_val in parameter_regex.findall(invoke_str):
            try:
                param_dict[param_name] = (
                    param_val
                    if is_string == "true"
                    or (
                        not is_string
                        and self._is_string_parameter(function_name, param_name)
                    )
                    else json.loads(param_val)
                )
            except json.JSONDecodeError:
                param_dict[param_name] = param_val
        return param_dict

    def _detect_format(self, text: str) -> bool:
        """Return True when DSML-prefixed format is detected, False for plain."""
        return f"{self._DSML}tool_calls>" in text

    def extract_tool_calls(
        self, model_output: str
    ) -> List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]]:
        """
        Extract tool calls from complete model output.

        Args:
            model_output: The complete output string from the model.

        Returns:
            A list of tuples where each tuple contains:
            - content: Raw content if parsing failed, None if successful
            - function_name: Name of the function to call
            - parameters: Function parameters as dict
        """
        # Detect which format is present
        has_any = any(tok in model_output for tok in self.tool_call_start_tokens)
        if not has_any:
            return self._recover_orphans(model_output)

        parameter_regex = self._parameter_regex

        results: List[Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]] = (
            []
        )

        content_start = 0
        for start, end, body in self._wrapped_blocks(model_output):
            invokes = [
                (head[1], body)
                for head, _, body, complete in self._invoke_frames(body)
                if complete
            ]
            if not invokes:
                continue
            content = model_output[content_start:start]
            if content:
                results.append((content, None, None))
            for invoke_name, invoke_content in invokes:
                param_dict = self._parse_invoke_params(
                    invoke_content, parameter_regex, invoke_name
                )
                results.append((None, invoke_name, param_dict))
            content_start = end

        if not results:
            return [(model_output, None, None)]

        content = model_output[content_start:]
        if content:
            results.append((content, None, None))
        return results

    def _wrapped_blocks(self, text: str):
        # A literal tool_calls closer inside a parameter value is not a
        # wrapper boundary, just as a literal invoke closer is not a call end.
        frames = [
            (head.start(), end)
            for head, end, _, complete in self._invoke_frames(text)
            if complete
        ]
        position = 0
        while (opening := self._wrapper_open.search(text, position)) is not None:
            closing = self._wrapper_close.search(text, opening.end())
            while closing is not None:
                enclosing_end = next(
                    (end for start, end in frames if start <= closing.start() < end),
                    None,
                )
                if enclosing_end is None:
                    break
                closing = self._wrapper_close.search(text, enclosing_end)
            if closing is None:
                return
            yield opening.start(), closing.end(), text[opening.end() : closing.start()]
            position = closing.end()

    def _orphan_allowed(self, text: str, match: re.Match) -> bool:
        # Do not promote examples in Markdown code fences or inline prose.
        before = text[: match.start()]
        if len(re.findall(r"(?m)^\s*(?:```|~~~)", before)) % 2:
            return False
        if before.rsplit("\n", 1)[-1].strip() and not re.search(
            r"</(?:｜DSML｜)?invoke>\s*$", before
        ):
            return False
        tools = request_tools_var.get()
        if tools is not None:
            return any(
                tool.get("function", tool).get("name") == match[1] for tool in tools
            )
        return False

    def _invoke_frames(self, text: str):
        """Scan provisional invokes; a wrapper closer cannot commit a call.

        State is local to this parse, because one parser serves concurrent
        requests. Parameter values are opaque until their closing tag: source
        code containing an invoke tag must not terminate the provisional call.
        """
        prefix = r"(?:｜DSML｜)?"
        opening = re.compile(rf'<{prefix}invoke\s+name="([^"]+)"\s*>')
        parameter = re.compile(
            rf'<{prefix}parameter\s+name="[^"]+"'
            rf'(?:\s+string="(?:true|false)")?\s*>'
        )
        parameter_end = re.compile(rf"</{prefix}parameter>")
        invoke_end = re.compile(rf"</{prefix}invoke>")
        position = 0
        while (head := opening.search(text, position)) is not None:
            cursor = head.end()
            # PROVISIONAL -> PARAMETER -> PROVISIONAL -> COMMIT.
            while True:
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                end = invoke_end.match(text, cursor)
                if end is not None:
                    yield head, end.end(), text[head.end() : cursor], True
                    position = end.end()
                    break
                param = parameter.match(text, cursor)
                if param is not None:
                    close = parameter_end.search(text, param.end())
                    if close is None:
                        yield head, len(text), "", False
                        return
                    cursor = close.end()
                    continue
                # An unfinished or malformed candidate is retained as text.
                # Never use an outer tool_calls closer as an invoke closer.
                end = invoke_end.search(text, cursor)
                if end is None:
                    yield head, len(text), "", False
                    return
                yield head, end.end(), "", False
                position = end.end()
                break

    def _recover_orphans(self, text: str) -> List[ToolCall]:
        results: List[ToolCall] = []
        position = 0
        for match, end, body, complete in self._invoke_frames(text):
            if not complete or not self._orphan_allowed(text, match):
                continue
            content = text[position : match.start()]
            if content:
                results.append((content, None, None))
            results.append(
                (
                    None,
                    match[1],
                    self._parse_invoke_params(body, self._parameter_regex, match[1]),
                )
            )
            position = end
            # Some responses append duplicate structural closing tags.
            tail = re.match(
                r"(?:\s*</(?:｜DSML｜)?(?:parameter|invoke|tool_calls)>)*",
                text[position:],
            )
            assert tail is not None
            position += tail.end()
        if position < len(text):
            results.append((text[position:], None, None))
        return results or [(text, None, None)]

    def _held_suffix_length(self, text: str, markers) -> int:
        held = self._partial_marker_length(text, markers)
        # A marker with one literal space cannot hold a header split across
        # arbitrary whitespace accepted by the structural scanner (\s+).
        pending = re.search(
            r'<(?:｜DSML｜)?invoke(?:\s+(?:n|na|nam|name(?:=(?:"[^>]*)?)?)?)?$',
            text,
        )
        return max(held, len(pending[0]) if pending else 0)

    @property
    def _closing_markers(self) -> Tuple[str, ...]:
        return tuple(
            f"</{prefix}{tag}>"
            for prefix in ("", "｜DSML｜")
            for tag in ("parameter", "invoke", "tool_calls")
        )

    def _stream_orphans(self, text: str, delta: str):
        markers = (
            '<invoke name="',
            '<｜DSML｜invoke name="',
            *self.tool_call_start_tokens,
        )
        starts = list(re.finditer(r'<(?:｜DSML｜)?invoke\s+name="', text))
        previous_end = len(text) - len(delta)
        if not starts:
            old = text[:previous_end]
            start = previous_end - self._held_suffix_length(old, markers)
            end = len(text) - self._held_suffix_length(text, markers)
            content = text[start:end]
            return (content, None, None) if content else None
        events: List[ToolCall] = []

        def emit_region(start: int, end: int, after_call: bool = False):
            region = text[start:end]
            if after_call:
                tail = re.match(
                    r"(?:\s*</(?:｜DSML｜)?(?:parameter|invoke|tool_calls)>)*", region
                )
                assert tail is not None
                start += tail.end()
            hold_markers = markers + self._closing_markers if after_call else markers
            old_end = min(previous_end, end)
            emitted = max(
                start,
                old_end - self._held_suffix_length(text[start:old_end], hold_markers),
            )
            safe_end = end - self._held_suffix_length(text[start:end], hold_markers)
            if safe_end > emitted:
                events.append((text[emitted:safe_end], None, None))

        position = 0
        after_call = False
        for head, end, body, complete in self._invoke_frames(text):
            emit_region(position, head.start(), after_call)
            if not complete and end == len(text):
                return events or None
            allowed = complete and self._orphan_allowed(text, head)
            if end > previous_end:
                if allowed:
                    events.append(
                        (
                            None,
                            head[1],
                            self._parse_invoke_params(
                                body, self._parameter_regex, head[1]
                            ),
                        )
                    )
                else:
                    events.append((text[head.start() : end], None, None))
            position = end
            after_call = allowed
        pending = next(
            (head.start() for head in starts if head.start() >= position), None
        )
        emit_region(position, pending if pending is not None else len(text), after_call)
        return events or None

    def flush_streaming_content(self, text: str) -> Optional[str]:
        if any(tok in text for tok in self.tool_call_start_tokens):
            return None
        frames = list(self._invoke_frames(text))
        if frames:
            head, end, _, complete = frames[-1]
            if not complete and end == len(text):
                return text[head.start() :]
        # A split opening marker may not yet have a complete function name.
        partial = re.search(r'<(?:｜DSML｜)?invoke\s+name="[^>]*$', text)
        if partial:
            return partial[0]
        markers = (
            '<invoke name="',
            '<｜DSML｜invoke name="',
            *self.tool_call_start_tokens,
        )
        # Use the same held markers as streaming. A suffix such as </par
        # is still ordinary text if generation ends before a full closer.
        if frames:
            head, end, _, complete = frames[-1]
            if complete and self._orphan_allowed(text, head):
                markers += self._closing_markers
                text = text[end:]
        held = self._held_suffix_length(text, markers)
        return text[-held:] if held else None

    def locate_argument_start(self, text: str) -> Optional[ArgumentStart]:
        dsml = self._detect_format(text)
        # Tolerant frames are buffered until a complete invoke is validated.
        # The shared incremental argument reader expects one exact dialect.
        other = "" if dsml else "｜DSML｜"
        if not any(tok in text for tok in self.tool_call_start_tokens):
            return None
        if re.search(rf"<{re.escape(other)}(?:invoke|parameter)\s", text):
            return None
        for parameter in re.finditer(r"<(?:｜DSML｜)?parameter\s+[^>]*>", text):
            if not re.search(r'\sstring="(?:true|false)"', parameter[0]):
                return None
        return dsml_argument_start(
            text,
            "tool_calls",
            self._DSML[1:] if dsml else "",
            lambda body: self._parse_invoke_params(body, self._parameter_regex),
        )

    def extract_tool_calls_streaming(
        self, previous_text: List[str], current_text: str, delta_text: str
    ) -> Optional[Union[ToolCall, List[ToolCall]]]:
        """
        Extract tool calls from streaming output.

        Buffers tokens until a complete invoke block is available, then
        parses and returns it in one shot. The parser is shared by
        concurrent requests, so progress is derived from this request's
        accumulated text and delta, never from instance state.

        Args:
            previous_text: Previous text chunks from the stream.
            current_text: Current accumulated text.
            delta_text: New text delta in this chunk.

        Returns:
            None if waiting for more data, a tuple of (content, function_name,
            parameters), or a list of them when one delta completes several
            invokes.
        """
        try:
            if not any(tok in current_text for tok in self.tool_call_start_tokens):
                return self._stream_orphans(current_text, delta_text)

            # Inside tool-call region: return the invokes whose closing tag
            # arrived in this delta
            parameter_regex = self._parameter_regex
            previous_length = len(current_text) - len(delta_text)
            # Every invoke ends with its closing tag, so none can complete in
            # this delta unless the tag does. Skip the regex scan until then.
            if not any(
                current_text.find(close, max(previous_length - len(close) + 1, 0)) != -1
                for close in ("</｜DSML｜invoke>", "</invoke>")
            ):
                return None
            results: List[ToolCall] = []
            for head, end, invoke_body, complete in self._invoke_frames(current_text):
                if complete and end > previous_length:
                    invoke_name = head[1]
                    param_dict = self._parse_invoke_params(
                        invoke_body, parameter_regex, invoke_name
                    )
                    results.append((None, invoke_name, param_dict))

            if not results:
                # Buffering — no invoke completed in this delta
                return None
            return results[0] if len(results) == 1 else results

        except Exception as e:
            logger.error("Error in DeepSeek V4 streaming tool call extraction: %s", e)
            raise
