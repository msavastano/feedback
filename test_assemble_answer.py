"""Unit test for joining a multi-part model answer. Run: python test_assemble_answer.py

Search grounding (Gemini) and web search citations (Claude) cut one answer
into several text parts, often mid-sentence. They used to be joined with a
blank line, which the chat rendered as a paragraph break after a comma:

    If that also fails,

    a majority of the Knesset members may request ...
"""
import os
from types import SimpleNamespace as NS

os.environ["AGENT_SWEEP"] = "0"
os.environ["AGENT_REFLECT"] = "0"
os.environ["AGENT_IMAGE_GEN"] = "0"

import agent  # noqa: E402


def _gemini_resp(*parts):
    return NS(candidates=[NS(content=NS(parts=list(parts)))], text=None)


def _text(t):
    return NS(text=t, executable_code=None, code_execution_result=None, function_call=None)


def test_gemini_split_text_joins_without_break():
    resp = _gemini_resp(
        _text("If that also fails,"),
        _text(" a majority of the Knesset members may request a candidate."),
    )
    assert agent._assemble_answer(resp) == (
        "If that also fails, a majority of the Knesset members may request a candidate."
    )


def test_gemini_empty_part_does_not_split_text():
    resp = _gemini_resp(_text("one,"), _text(""), _text(" two"))
    assert agent._assemble_answer(resp) == "one, two"


def test_gemini_code_blocks_keep_blank_lines():
    code = NS(text=None, executable_code=NS(code="print(1)"),
              code_execution_result=None, function_call=None)
    result = NS(text=None, executable_code=None,
                code_execution_result=NS(output="1"), function_call=None)
    resp = _gemini_resp(_text("Running it:"), code, result, _text("Done."))
    assert agent._assemble_answer(resp) == (
        "Running it:\n\n```python\nprint(1)\n```\n\n```\n1\n```\n\nDone."
    )


def test_claude_citation_blocks_join_without_break():
    class FakeMessages:
        def create(self, **kw):
            return NS(
                stop_reason="end_turn",
                usage=NS(input_tokens=1, output_tokens=1,
                         cache_read_input_tokens=0, cache_creation_input_tokens=0,
                         server_tool_use=None),
                content=[
                    NS(type="server_tool_use"),
                    NS(type="web_search_tool_result"),
                    NS(type="text", text="If that also fails,"),
                    NS(type="text", text=" a majority of the Knesset members may request it."),
                ],
            )

    clients = agent.Clients(gemini=None, anthropic=NS(messages=FakeMessages()))
    text, _ = agent._chat_respond(
        clients, agent.CLAUDE_MODEL, "sys", [], "q", [],
        allow_code=False, allow_search=True, thinking_level=None, label="t",
    )
    assert text == "If that also fails, a majority of the Knesset members may request it."


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
