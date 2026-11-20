"""Request-owned incremental scanner for DeepSeek V4 tool calls."""

import json
import re
from typing import Any, Dict, Optional, Tuple

from .argument_stream import ArgumentStart

_PREFIX = r"(?:｜DSML｜)?"
_INVOKE = re.compile(rf'<{_PREFIX}invoke\s+name="([^"]+)"\s*>')
_PARAMETER = re.compile(
    rf'<{_PREFIX}parameter\s+name="([^"]+)"' rf'(?:\s+string="(true|false)")?\s*>'
)
_WRAPPERS = ("<tool_calls>", "<｜DSML｜tool_calls>")
_INVOKES = ("<invoke", "<｜DSML｜invoke")
_PARAMETERS = ("<parameter", "<｜DSML｜parameter")
_PARAM_ENDS = ("</parameter>", "</｜DSML｜parameter>")
_INVOKE_ENDS = ("</invoke>", "</｜DSML｜invoke>")
_WRAPPER_ENDS = ("</tool_calls>", "</｜DSML｜tool_calls>")
_CLOSERS = _PARAM_ENDS + _INVOKE_ENDS + _WRAPPER_ENDS


class DeepseekV4StreamState:
    """Offsets belong to one request, never to the shared parser instance.

    The caller already owns the cumulative text. Keep offsets into that text
    instead of copying an ever-growing provisional invoke on every delta.
    """

    def __init__(self):
        self.position = 0
        self.search = 0
        self.seen = 0
        self.phase = "text"
        self.wrapped = False
        self.after_call = False
        self.call_start = 0
        self.body_start = 0
        self.name = ""
        self.arguments: Dict[str, Any] = {}
        self.allowed = False
        self.valid = False
        self.orphan = False
        self.key = ""
        self.is_string: Optional[str] = None
        self.value_start = 0
        self.argument_start: Optional[ArgumentStart] = None
        self.finished = False

    @staticmethod
    def _partial(fragment, markers):
        return bool(fragment) and any(
            len(fragment) < len(marker) and marker.startswith(fragment)
            for marker in markers
        )

    def feed(self, parser, text: str, delta: str):
        if self.finished:
            return None
        if len(text) - len(delta) != self.seen:
            raise ValueError("DeepSeek V4 stream state must belong to one request")
        self.seen = len(text)
        events = []

        def content(start, end):
            if end > start:
                value = text[start:end]
                # Only an immediately adjacent structural tail may be absorbed.
                # Once prose resumes, later closing tags are literal content.
                if value.strip():
                    self.after_call = False
                events.append((value, None, None))

        while self.position < len(text):
            if self.phase == "text":
                angle = text.find("<", self.position)
                if angle == -1:
                    if not self.wrapped:
                        content(self.position, len(text))
                    self.position = len(text)
                    break
                if not self.wrapped:
                    content(self.position, angle)
                self.position = angle
                fragment = text[angle : angle + 32]
                markers: Tuple[str, ...] = _WRAPPERS + _INVOKES
                if self.wrapped or self.after_call:
                    markers += _CLOSERS
                if self._partial(fragment, markers):
                    break
                wrapper = next((m for m in _WRAPPERS if fragment.startswith(m)), None)
                if wrapper:
                    self.wrapped = True
                    self.position += len(wrapper)
                    continue
                if any(fragment.startswith(m) for m in _INVOKES):
                    self.call_start = angle
                    self.orphan = not self.wrapped
                    self.phase = "invoke_header"
                    self.search = angle
                    continue
                closer = next((m for m in _CLOSERS if fragment.startswith(m)), None)
                if closer and (self.wrapped or self.after_call):
                    if closer in _WRAPPER_ENDS:
                        self.wrapped = False
                    self.position += len(closer)
                    continue
                if not self.wrapped:
                    content(angle, angle + 1)
                self.position += 1
                continue

            if self.phase in ("invoke_header", "parameter_header"):
                end = text.find(">", self.search)
                if end == -1:
                    self.search = len(text)
                    break
                pattern = _INVOKE if self.phase == "invoke_header" else _PARAMETER
                match = pattern.fullmatch(text, self.position, end + 1)
                if match is None:
                    if self.phase == "invoke_header":
                        if not self.wrapped:
                            content(self.call_start, end + 1)
                        self.phase = "text"
                    else:
                        self.valid = False
                        self.phase = "broken"
                    self.position = end + 1
                    continue
                if self.phase == "invoke_header":
                    self.name = match[1]
                    self.body_start = end + 1
                    self.arguments = {}
                    self.valid = True
                    self.allowed = not self.orphan or parser._orphan_allowed(
                        text, match
                    )
                    self.argument_start = None
                    self.phase = "body"
                else:
                    self.key, self.is_string = match[1], match[2]
                    self.value_start = end + 1
                    self.phase = "parameter"
                    if (
                        not self.orphan
                        and self.is_string == "true"
                        and self.key not in self.arguments
                    ):
                        prefix_args = {**self.arguments, self.key: ""}
                        self.argument_start = ArgumentStart(
                            self.name,
                            self.value_start,
                            json.dumps(prefix_args, ensure_ascii=False)[:-2],
                            _PARAM_ENDS[0],
                            call_position=self.body_start,
                            terminators=(_PARAM_ENDS[1],),
                        )
                self.position = end + 1
                self.search = self.position
                continue

            if self.phase == "parameter":
                closing = text.find("</", self.search)
                if closing == -1:
                    self.search = max(self.value_start, len(text) - 1)
                    break
                fragment = text[closing : closing + 32]
                if self._partial(fragment, _PARAM_ENDS):
                    self.search = closing
                    break
                marker = next((m for m in _PARAM_ENDS if fragment.startswith(m)), None)
                if marker is None:
                    self.search = closing + 2
                    continue
                value = text[self.value_start : closing]
                if self.is_string == "true" or (
                    not self.is_string
                    and parser._is_string_parameter(self.name, self.key)
                ):
                    self.arguments[self.key] = value
                else:
                    try:
                        self.arguments[self.key] = json.loads(value)
                    except json.JSONDecodeError:
                        self.arguments[self.key] = value
                self.argument_start = None
                self.position = closing + len(marker)
                self.phase = "body"
                continue

            # A provisional invocation commits only on its own closer.
            if self.phase == "body":
                while self.position < len(text) and text[self.position].isspace():
                    self.position += 1
                if self.position == len(text):
                    break
                fragment = text[self.position : self.position + 32]
                if self._partial(fragment, _PARAMETERS + _INVOKE_ENDS + _WRAPPER_ENDS):
                    break
                if any(fragment.startswith(m) for m in _PARAMETERS):
                    self.phase = "parameter_header"
                    self.search = self.position
                    continue
                marker = next((m for m in _INVOKE_ENDS if fragment.startswith(m)), None)
                if marker:
                    self.position += len(marker)
                    accepted = self.valid and self.allowed
                    if accepted:
                        events.append((None, self.name, self.arguments))
                    elif self.orphan:
                        content(self.call_start, self.position)
                    self.after_call = accepted
                    self.argument_start = None
                    self.phase = "text"
                    continue
                self.valid = False
                self.argument_start = None
                self.phase = "broken"
                self.search = self.position
                continue

            if self.phase == "broken":
                closing = text.find("</", self.search)
                if closing == -1:
                    self.search = max(self.position, len(text) - 1)
                    break
                fragment = text[closing : closing + 32]
                if self._partial(fragment, _INVOKE_ENDS + _WRAPPER_ENDS):
                    self.search = closing
                    break
                marker = next(
                    (m for m in _INVOKE_ENDS + _WRAPPER_ENDS if fragment.startswith(m)),
                    None,
                )
                if marker is None:
                    self.search = closing + 2
                    continue
                self.position = closing + len(marker)
                if self.orphan:
                    content(self.call_start, self.position)
                if marker in _WRAPPER_ENDS:
                    self.wrapped = False
                self.phase = "text"
                self.after_call = False
        return events or None

    def finish(self, text: str):
        if self.finished:
            return None
        self.finished = True
        self.argument_start = None
        if self.phase != "text":
            return text[self.call_start :] if self.orphan else None
        if not self.wrapped and self.position < len(text):
            return text[self.position :]
        return None
