from __future__ import annotations

import unittest
from unittest.mock import Mock, call, patch

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)

from actionweaver.actions import Action, ActionHandlers
from actionweaver.llms.azure.chat import ChatCompletion as AzureChatCompletion


class TestAzureChatCompletion(unittest.TestCase):
    def generate_mock_function_call_response(self, name, arguments):
        return ChatCompletion(
            **{
                "id": "chatcmpl-8J02pR3nTveTRRgDsAP94HpG2pyi9",
                "choices": [
                    {
                        "finish_reason": "function_call",
                        "index": 0,
                        "message": {
                            "content": None,
                            "role": "assistant",
                            "function_call": {
                                "arguments": arguments,
                                "name": name,
                            },
                            "tool_calls": None,
                        },
                    }
                ],
                "created": 1699539095,
                "model": "gpt-3.5-turbo-0613",
                "object": "chat.completion",
                "system_fingerprint": None,
                "usage": {
                    "completion_tokens": 18,
                    "prompt_tokens": 83,
                    "total_tokens": 101,
                },
            }
        )

    def generate_mock_message_response(self, content):
        return ChatCompletion(
            **{
                "id": "chatcmpl-8IzsbGIxAwpvBncWoh3Hy4jFCqo35",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": content,
                            "role": "assistant",
                            "function_call": None,
                            "tool_calls": None,
                        },
                    }
                ],
                "created": 1699538461,
                "model": "gpt-3.5-turbo-0613",
                "object": "chat.completion",
                "system_fingerprint": None,
                "usage": {
                    "completion_tokens": 9,
                    "prompt_tokens": 19,
                    "total_tokens": 28,
                },
            }
        )

    @patch("openai.resources.chat.Completions.create")
    def test_create_message(self, mock_create):
        # Create an instance of OpenAIChatCompletion
        chat_completion = AzureChatCompletion(
            model="test",
            azure_endpoint="AZURE_OPENAI_ENDPOINT",
            api_key="AZURE_OPENAI_KEY",
            api_version="2023-10-01-preview",
        )

        # Define the expected functions arguments and return values in the API call
        expected_functions_and_results = [
            (
                None,
                self.generate_mock_message_response("Hello! what can I do for you"),
            ),
        ]

        # Set the return values of the mock
        mock_create.side_effect = [
            expected_result for _, expected_result in expected_functions_and_results
        ]

        # When
        messages = [{"role": "user", "content": "Hi!"}]
        response = chat_completion.create(messages=messages)

        # Then
        mock_create.assert_called_once()
        self.assertFalse("functions" in mock_create.call_args_list[0].kwargs)
        self.assertFalse("function_call" in mock_create.call_args_list[0].kwargs)
        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "Hi!"},
            ],
        )
        self.assertEqual(response, "Hello! what can I do for you")

    @patch("openai.resources.chat.Completions.create")
    def test_create_with_functions1(self, mock_create):
        def mock_method(text: str):
            """mock method"""
            return text

        actions = [
            Action("action1", mock_method).build_pydantic_model_cls(),
        ]
        chat_completion = AzureChatCompletion(
            model="test",
            azure_endpoint="AZURE_OPENAI_ENDPOINT",
            api_key="AZURE_OPENAI_KEY",
            api_version="2023-10-01-preview",
        )
        # Define the expected functions arguments and return values in the API call
        expected_functions_and_results = [
            (
                {"functions": ["action1"], "function_call": "auto"},
                self.generate_mock_function_call_response(
                    "action1", '{\n  "text": "echo1"\n}'
                ),
            ),
            (
                {"functions": ["action1"], "function_call": "auto"},
                self.generate_mock_message_response("last message"),
            ),
        ]

        # Set the return values of the mock
        mock_create.side_effect = [
            expected_result for _, expected_result in expected_functions_and_results
        ]

        # When
        messages = [{"role": "user", "content": "Hi!"}]
        response = chat_completion.create(messages=messages, actions=actions)

        # Then
        # Use a loop to iterate over expected calls and assert function arguments in the API call
        for i, actual_call in enumerate(mock_create.call_args_list):
            if "functions" in actual_call.kwargs:
                self.assertEqual(
                    [func["name"] for func in actual_call.kwargs["functions"]],
                    expected_functions_and_results[i][0]["functions"],
                )
                self.assertEqual(
                    actual_call.kwargs["function_call"],
                    expected_functions_and_results[i][0]["function_call"],
                )
            else:
                self.assertFalse("functions" in expected_functions_and_results[i][0])
                self.assertFalse(
                    "function_call" in expected_functions_and_results[i][0]
                )

        self.assertEqual(
            messages,
            [
                {"content": "Hi!", "role": "user"},
                {
                    "content": None,
                    "function_call": {
                        "arguments": '{\n  "text": "echo1"\n}',
                        "name": "action1",
                    },
                    "role": "assistant",
                },
                {"content": "echo1", "name": "action1", "role": "function"},
            ],
        )
        self.assertEqual(response, "last message")

    @patch("openai.resources.chat.Completions.create")
    def test_create_with_functions2(self, mock_create):
        def mock_method(text: str):
            """mock method"""
            return text

        # Create an instance of OpenAIChatCompletion with action handlers
        actions = [
            Action(
                "action1",
                mock_method,
            ).build_pydantic_model_cls(),
            Action(
                "action2",
                mock_method,
            ).build_pydantic_model_cls(),
            Action(
                "action3",
                mock_method,
            ).build_pydantic_model_cls(),
            Action(
                "action4",
                mock_method,
            ).build_pydantic_model_cls(),
        ]
        chat_completion = AzureChatCompletion(
            model="test",
            azure_endpoint="AZURE_OPENAI_ENDPOINT",
            api_key="AZURE_OPENAI_KEY",
            api_version="2023-10-01-preview",
        )

        # Define the expected functions arguments and return values in the API call
        expected_functions_and_results = [
            (
                {
                    "functions": ["action1", "action2", "action3", "action4"],
                    "function_call": "auto",
                },
                self.generate_mock_function_call_response(
                    "action1", '{\n  "text": "echo1"\n}'
                ),
            ),
            (
                {
                    "functions": ["action2", "action3"],
                    "function_call": "auto",
                },
                self.generate_mock_function_call_response(
                    "action2", '{\n  "text": "echo2"\n}'
                ),
            ),
            (
                {"functions": ["action3"], "function_call": {"name": "action3"}},
                self.generate_mock_function_call_response(
                    "action3", '{\n  "text": "echo3"\n}'
                ),
            ),
            (
                {"functions": ["action4"], "function_call": {"name": "action4"}},
                self.generate_mock_function_call_response(
                    "action4", '{\n  "text": "echo4"\n}'
                ),
            ),
            (
                {},
                self.generate_mock_message_response("last message"),
            ),
        ]

        # Set the return values of the mock
        mock_create.side_effect = [
            expected_result for _, expected_result in expected_functions_and_results
        ]

        # When
        messages = [{"role": "user", "content": "Hi!"}]
        response = chat_completion.create(
            messages=messages,
            actions=actions,
            orch={
                actions[0]: [actions[1], actions[2]],
                actions[1]: actions[2],
                actions[2]: actions[3],
                actions[3]: None,
            },
        )

        # Then
        # Use a loop to iterate over expected calls and assert function arguments in the API call
        for i, actual_call in enumerate(mock_create.call_args_list):
            if "functions" in actual_call.kwargs:
                self.assertEqual(
                    [func["name"] for func in actual_call.kwargs["functions"]],
                    expected_functions_and_results[i][0]["functions"],
                )
                self.assertEqual(
                    actual_call.kwargs["function_call"],
                    expected_functions_and_results[i][0]["function_call"],
                )
            else:
                self.assertFalse("functions" in expected_functions_and_results[i][0])
                self.assertFalse(
                    "function_call" in expected_functions_and_results[i][0]
                )

        self.assertEqual(
            messages,
            [
                {"content": "Hi!", "role": "user"},
                {
                    "content": None,
                    "function_call": {
                        "arguments": '{\n  "text": "echo1"\n}',
                        "name": "action1",
                    },
                    "role": "assistant",
                },
                {"content": "echo1", "name": "action1", "role": "function"},
                {
                    "content": None,
                    "function_call": {
                        "arguments": '{\n  "text": "echo2"\n}',
                        "name": "action2",
                    },
                    "role": "assistant",
                },
                {"content": "echo2", "name": "action2", "role": "function"},
                {
                    "content": None,
                    "function_call": {
                        "arguments": '{\n  "text": "echo3"\n}',
                        "name": "action3",
                    },
                    "role": "assistant",
                },
                {"content": "echo3", "name": "action3", "role": "function"},
                {
                    "content": None,
                    "function_call": {
                        "arguments": '{\n  "text": "echo4"\n}',
                        "name": "action4",
                    },
                    "role": "assistant",
                },
                {"content": "echo4", "name": "action4", "role": "function"},
            ],
        )
        self.assertEqual(response, "last message")


if __name__ == "__main__":
    unittest.main()
