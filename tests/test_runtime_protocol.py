"""CPU unit tests for the runtime protocol layer (no model, no GPU)."""

import unittest

from yuzhu_lab.runtime.protocol import (
    AnthropicMessagesRequest,
    ParsedToolCall,
    anthropic_messages_to_internal,
    openai_messages_to_internal,
    parse_tool_calls,
    stop_list,
)


class ParseToolCallsTest(unittest.TestCase):
    def test_plain_text_has_no_calls(self):
        text, calls = parse_tool_calls("Just a normal answer.")
        self.assertEqual(text, "Just a normal answer.")
        self.assertEqual(calls, [])

    def test_single_call_with_typed_params(self):
        raw = (
            'Let me read it.\n<function name="read_file">'
            '<param name="path">src/app.py</param>'
            '<param name="line">12</param>'
            '<param name="strict">true</param>'
            "</function>"
        )
        text, calls = parse_tool_calls(raw)
        self.assertEqual(text, "Let me read it.")
        self.assertEqual(
            calls,
            [ParsedToolCall(name="read_file", arguments={"path": "src/app.py", "line": 12, "strict": True})],
        )

    def test_cdata_preserves_multiline_value(self):
        raw = (
            '<function name="write_file"><param name="path">a.py</param>'
            '<param name="content"><![CDATA[def f():\n    return 1\n]]></param></function>'
        )
        text, calls = parse_tool_calls(raw)
        self.assertEqual(text, "")
        self.assertEqual(calls[0].arguments["content"], "def f():\n    return 1\n")

    def test_multiple_calls_and_text(self):
        raw = (
            "Two steps:\n"
            '<function name="a"><param name="x">1</param></function>\n'
            '<function name="b"><param name="y">2</param></function>\n'
            "done"
        )
        text, calls = parse_tool_calls(raw)
        self.assertEqual([call.name for call in calls], ["a", "b"])
        self.assertIn("Two steps:", text)
        self.assertIn("done", text)
        self.assertNotIn("<function", text)

    def test_malformed_function_stays_in_text(self):
        raw = '<function name="x"><param name="p">v</param>'
        text, calls = parse_tool_calls(raw)
        self.assertEqual(text, raw)
        self.assertEqual(calls, [])


class AnthropicConversionTest(unittest.TestCase):
    def test_tool_use_and_tool_result_round_trip(self):
        request = AnthropicMessagesRequest(
            system="sys",
            messages=[
                {"role": "user", "content": "read a.py"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "reading"},
                        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "print(1)"}
                    ],
                },
            ],
            max_tokens=32,
        )
        internal = anthropic_messages_to_internal(request)
        self.assertEqual(internal[0], {"role": "system", "content": "sys"})
        self.assertEqual(internal[1], {"role": "user", "content": "read a.py"})
        self.assertEqual(internal[2]["tool_calls"][0]["name"], "read_file")
        self.assertEqual(internal[3], {"role": "tool", "content": "print(1)"})

    def test_image_block_rejected(self):
        request = AnthropicMessagesRequest(
            messages=[
                {"role": "user", "content": [{"type": "image", "source": {"type": "base64"}}]}
            ],
            max_tokens=8,
        )
        with self.assertRaises(ValueError):
            anthropic_messages_to_internal(request)

    def test_openai_tool_message(self):
        internal = openai_messages_to_internal(
            [
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": "result", "tool_call_id": "x"},
            ]
        )
        self.assertEqual(internal[1], {"role": "tool", "content": "result"})

    def test_stop_list(self):
        self.assertEqual(stop_list("x"), ["x"])
        self.assertEqual(stop_list(["a", "", "b"]), ["a", "b"])
        self.assertEqual(stop_list(None), [])


if __name__ == "__main__":
    unittest.main()