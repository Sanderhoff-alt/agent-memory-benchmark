import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from memory_bench.llm.anthropic import AnthropicLLM, _parse_custom_headers
from memory_bench.llm.base import Schema


_SCHEMA = Schema(
    properties={
        "answer": {"type": "string"},
        "correct": {"type": "boolean"},
    },
    required=["answer", "correct"],
)


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def _tool_response(payload):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", name="structured_response", input=payload),
        ]
    )


def _text_response(text):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text=text),
        ]
    )


def _llm_with_responses(responses):
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm._client = _FakeClient(responses)
    llm._model = "test-model"
    return llm


class AnthropicLLMTests(unittest.TestCase):
    def test_default_model_is_sonnet_4_6(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            llm = AnthropicLLM()

        self.assertEqual(llm.model_id, "anthropic:claude-sonnet-4-6")

    def test_generate_returns_forced_tool_payload(self):
        llm = _llm_with_responses([_tool_response({"answer": "yes", "correct": True})])

        result = llm.generate("prompt", _SCHEMA)

        self.assertEqual(result, {"answer": "yes", "correct": True})
        call = llm._client.messages.calls[0]
        self.assertEqual(call["tool_choice"], {"type": "tool", "name": "structured_response"})
        self.assertEqual(call["tools"][0]["input_schema"]["required"], ["answer", "correct"])

    def test_generate_falls_back_to_text_json_when_tools_are_unsupported(self):
        llm = _llm_with_responses(
            [
                RuntimeError("tool_choice is not supported by this endpoint"),
                _text_response('{"answer": "fallback", "correct": false}'),
            ]
        )

        result = llm.generate("prompt", _SCHEMA)

        self.assertEqual(result, {"answer": "fallback", "correct": False})
        self.assertEqual(len(llm._client.messages.calls), 2)
        self.assertIn("system", llm._client.messages.calls[1])

    def test_generate_parses_fenced_json_text_fallback(self):
        llm = _llm_with_responses(
            [
                RuntimeError("tools unsupported"),
                _text_response('```json\n{"answer": "inside fence", "correct": true}\n```'),
            ]
        )

        result = llm.generate("prompt", _SCHEMA)

        self.assertEqual(result, {"answer": "inside fence", "correct": True})

    def test_retryable_status_error_accepts_gateway_400_with_upstream_500(self):
        class GatewayError(Exception):
            status_code = 400

            def __str__(self):
                return '{"error": {"code": "500", "message": "upstream failed"}}'

        self.assertTrue(AnthropicLLM._is_retryable_status_error(GatewayError()))

    def test_retryable_status_error_rejects_plain_400(self):
        class BadRequest(Exception):
            status_code = 400

            def __str__(self):
                return "invalid request"

        self.assertFalse(AnthropicLLM._is_retryable_status_error(BadRequest()))

    def test_parse_custom_headers_empty(self):
        self.assertIsNone(_parse_custom_headers(None))
        self.assertIsNone(_parse_custom_headers(""))
        self.assertIsNone(_parse_custom_headers("   "))

    def test_parse_custom_headers_valid_json(self):
        raw = '{"X-Custom": "val", "num": 123}'
        headers = _parse_custom_headers(raw)
        self.assertEqual(headers, {"X-Custom": "val", "num": "123"})

    def test_parse_custom_headers_nested_object_serialized(self):
        raw = '{"X-Nested-Header": {"k1": "v1", "k2": "v2"}}'
        headers = _parse_custom_headers(raw)
        self.assertEqual(
            headers,
            {"X-Nested-Header": '{"k1":"v1","k2":"v2"}'},
        )

    def test_parse_custom_headers_invalid_json(self):
        with self.assertRaises(ValueError):
            _parse_custom_headers("not-json")

    def test_parse_custom_headers_non_dict_json(self):
        with self.assertRaises(ValueError):
            _parse_custom_headers('["not", "a", "dict"]')

    def test_init_with_custom_headers_env(self):
        headers_json = '{"X-Trace-Id": "mock-trace-123", "X-Payload": {"env": "test"}}'
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test-key",
                "ANTHROPIC_CUSTOM_HEADERS": headers_json,
            },
            clear=False,
        ), patch("anthropic.Anthropic") as mock_anthropic:
            llm = AnthropicLLM()
            mock_anthropic.assert_called_once_with(
                api_key="test-key",
                base_url=None,
                default_headers={
                    "X-Trace-Id": "mock-trace-123",
                    "X-Payload": '{"env":"test"}',
                },
                max_retries=0,
            )

    def test_init_with_explicit_default_headers_override(self):
        headers_json = '{"X-Header": "env-val", "X-Keep": "1"}'
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test-key",
                "ANTHROPIC_CUSTOM_HEADERS": headers_json,
            },
            clear=False,
        ), patch("anthropic.Anthropic") as mock_anthropic:
            llm = AnthropicLLM(default_headers={"X-Header": "override-val", "X-Extra": "2"})
            mock_anthropic.assert_called_once_with(
                api_key="test-key",
                base_url=None,
                default_headers={
                    "X-Header": "override-val",
                    "X-Keep": "1",
                    "X-Extra": "2",
                },
                max_retries=0,
            )


if __name__ == "__main__":
    unittest.main()
