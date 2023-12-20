from __future__ import annotations

import unittest
from unittest.mock import Mock, call, patch

from openai.types.chat.chat_completion import ChatCompletion, Choice, CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from actionweaver.actions import Action
from actionweaver.llms.openai.tools.chat import OpenAIChatCompletion


class TestOpenAIChatCompletion(unittest.TestCase):
    def generate_multiple_mock_function_call_response(self, name, arguments):
        return ChatCompletion(
            **{
                "id": "chatcmpl-8WCZDJ12zHTvK8YltBeeDN0NGn0PI",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "index": 0,
                        "message": {
                            "content": None,
                            "role": "assistant",
                            "function_call": None,
                            "tool_calls": [
                                {
                                    "id": "call_70TwJYHkDSs80tcnF5lt8TQR",
                                    "function": {
                                        "arguments": argument,
                                        "name": name,
                                    },
                                    "type": "function",
                                }
                                for name, argument in zip(names, arguments)
                            ],
                        },
                        "logprobs": None,
                    }
                ],
                "created": 1702685495,
                "model": "gpt-3.5-turbo-1106",
                "object": "chat.completion",
                "system_fingerprint": "fp_772e8125bb",
                "usage": {
                    "completion_tokens": 70,
                    "prompt_tokens": 130,
                    "total_tokens": 200,
                },
            }
        )

    def generate_single_mock_function_call_response(self, name, arguments):
        return ChatCompletion(
            **{
                "id": "chatcmpl-8WCZDJ12zHTvK8YltBeeDN0NGn0PI",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "index": 0,
                        "message": {
                            "content": None,
                            "role": "assistant",
                            "function_call": None,
                            "tool_calls": [
                                {
                                    "id": "call_70TwJYHkDSs80tcnF5lt8TQR",
                                    "function": {
                                        "arguments": arguments,
                                        "name": name,
                                    },
                                    "type": "function",
                                },
                            ],
                        },
                        "logprobs": None,
                    }
                ],
                "created": 1702685495,
                "model": "gpt-3.5-turbo-1106",
                "object": "chat.completion",
                "system_fingerprint": "fp_772e8125bb",
                "usage": {
                    "completion_tokens": 70,
                    "prompt_tokens": 130,
                    "total_tokens": 200,
                },
            }
        )

    def generate_mock_message_response(self, content):
        return ChatCompletion(
            **{
                "id": "chatcmpl-8WCdHNVdrYkU8cir7xYcji02Lenuw",
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
                        "logprobs": None,
                    }
                ],
                "created": 1702685747,
                "model": "gpt-3.5-turbo-1106",
                "object": "chat.completion",
                "system_fingerprint": "fp_772e8125bb",
                "usage": {
                    "completion_tokens": 35,
                    "prompt_tokens": 27,
                    "total_tokens": 62,
                },
            }
        )

    ### Tests for patching OpenAI client

    @patch("openai.OpenAI")
    def test_patched_create_message(self, mock_openai):
        client = mock_openai()
        mock_create = client.chat.completions.create

        client = OpenAIChatCompletion.patch(client)

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
        response = client.chat.completions.create(model="test", messages=messages)

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
        self.assertEqual(
            response,
            ChatCompletion(
                id="chatcmpl-8WCdHNVdrYkU8cir7xYcji02Lenuw",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        logprobs=None,
                        message=ChatCompletionMessage(
                            content="Hello! what can I do for you",
                            role="assistant",
                            function_call=None,
                            tool_calls=None,
                        ),
                    )
                ],
                created=1702685747,
                model="gpt-3.5-turbo-1106",
                object="chat.completion",
                system_fingerprint="fp_772e8125bb",
                usage=CompletionUsage(
                    completion_tokens=35, prompt_tokens=27, total_tokens=62
                ),
            ),
        )

    @patch("openai.OpenAI")
    def test_patched_create_with_single_function(self, mock_openai):
        client = mock_openai()
        mock_create = client.chat.completions.create
        client = OpenAIChatCompletion.patch(client)

        def mock_method(text: str):
            """mock method"""
            return text

        actions = [
            Action("action1", mock_method).build_pydantic_model_cls(),
        ]

        # Define the expected functions arguments and return values in the API call
        expected_functions_and_results = [
            (
                {"tools": ["action1"], "tool_choice": "auto"},
                self.generate_single_mock_function_call_response(
                    "action1", '{\n  "text": "echo1"\n}'
                ),
            ),
            (
                {"tools": ["action1"], "tool_choice": "auto"},
                self.generate_mock_message_response("last message"),
            ),
        ]

        # Set the return values of the mock
        mock_create.side_effect = [
            expected_result for _, expected_result in expected_functions_and_results
        ]

        # When
        messages = [{"role": "user", "content": "Hi!"}]
        response = client.chat.completions.create(
            model="test", messages=messages, actions=actions
        )

        # Then
        # Use a loop to iterate over expected calls and assert function arguments in the API call
        for i, actual_call in enumerate(mock_create.call_args_list):
            if "tools" in actual_call.kwargs:
                self.assertEqual(
                    [tool["function"]["name"] for tool in actual_call.kwargs["tools"]],
                    expected_functions_and_results[i][0]["tools"],
                )
                self.assertEqual(
                    actual_call.kwargs["tool_choice"],
                    expected_functions_and_results[i][0]["tool_choice"],
                )
            else:
                self.assertFalse("tools" in expected_functions_and_results[i][0])
                self.assertFalse("tool_choice" in expected_functions_and_results[i][0])

        self.assertEqual(
            messages,
            [
                {"content": "Hi!", "role": "user"},
                ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_70TwJYHkDSs80tcnF5lt8TQR",
                            function=Function(
                                arguments='{\n  "text": "echo1"\n}', name="action1"
                            ),
                            type="function",
                        )
                    ],
                ),
                {
                    "content": "echo1",
                    "name": "action1",
                    "role": "tool",
                    "tool_call_id": "call_70TwJYHkDSs80tcnF5lt8TQR",
                },
            ],
        )

        self.assertEqual(
            response,
            ChatCompletion(
                id="chatcmpl-8WCdHNVdrYkU8cir7xYcji02Lenuw",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        logprobs=None,
                        message=ChatCompletionMessage(
                            content="last message",
                            role="assistant",
                            function_call=None,
                            tool_calls=None,
                        ),
                    )
                ],
                created=1702685747,
                model="gpt-3.5-turbo-1106",
                object="chat.completion",
                system_fingerprint="fp_772e8125bb",
                usage=CompletionUsage(
                    completion_tokens=35, prompt_tokens=27, total_tokens=62
                ),
            ),
        )

    @patch("openai.OpenAI")
    def test_patched_create_with_single_function_orchestration(self, mock_openai):
        client = mock_openai()
        mock_create = client.chat.completions.create
        client = OpenAIChatCompletion.patch(client)

        def mock_method(text: str):
            """mock method"""
            return text

        actions = [
            Action("action1", mock_method).build_pydantic_model_cls(),
            Action("action2", mock_method).build_pydantic_model_cls(),
        ]

        # Define the expected functions arguments and return values in the API call
        expected_functions_and_results = [
            (
                {"tools": ["action1"], "tool_choice": "auto"},
                self.generate_single_mock_function_call_response(
                    "action1", '{\n  "text": "echo1"\n}'
                ),
            ),
            (
                {
                    "tools": ["action2"],
                    "tool_choice": {
                        "function": {"name": "action2"},
                        "type": "function",
                    },
                },
                self.generate_single_mock_function_call_response(
                    "action2", '{\n  "text": "echo2"\n}'
                ),
            ),
            (
                {"tools": ["action1"], "tool_choice": "auto"},
                self.generate_mock_message_response("last message"),
            ),
        ]

        # Set the return values of the mock
        mock_create.side_effect = [
            expected_result for _, expected_result in expected_functions_and_results
        ]

        # When
        messages = [{"role": "user", "content": "Hi!"}]
        response = client.chat.completions.create(
            model="test",
            messages=messages,
            actions=[actions[0]],
            orch={actions[0]: actions[1]},
        )

        # Then
        # Use a loop to iterate over expected calls and assert function arguments in the API call
        for i, actual_call in enumerate(mock_create.call_args_list):
            if "tools" in actual_call.kwargs:
                self.assertEqual(
                    [tool["function"]["name"] for tool in actual_call.kwargs["tools"]],
                    expected_functions_and_results[i][0]["tools"],
                )
                self.assertEqual(
                    actual_call.kwargs["tool_choice"],
                    expected_functions_and_results[i][0]["tool_choice"],
                )
            else:
                self.assertFalse("tools" in expected_functions_and_results[i][0])
                self.assertFalse("tool_choice" in expected_functions_and_results[i][0])

        self.assertEqual(
            messages,
            [
                {"content": "Hi!", "role": "user"},
                ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_70TwJYHkDSs80tcnF5lt8TQR",
                            function=Function(
                                arguments='{\n  "text": "echo1"\n}', name="action1"
                            ),
                            type="function",
                        )
                    ],
                ),
                {
                    "content": "echo1",
                    "name": "action1",
                    "role": "tool",
                    "tool_call_id": "call_70TwJYHkDSs80tcnF5lt8TQR",
                },
                ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_70TwJYHkDSs80tcnF5lt8TQR",
                            function=Function(
                                arguments='{\n  "text": "echo2"\n}', name="action2"
                            ),
                            type="function",
                        )
                    ],
                ),
                {
                    "content": "echo2",
                    "name": "action2",
                    "role": "tool",
                    "tool_call_id": "call_70TwJYHkDSs80tcnF5lt8TQR",
                },
            ],
        )

        self.assertEqual(
            response,
            ChatCompletion(
                id="chatcmpl-8WCdHNVdrYkU8cir7xYcji02Lenuw",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        logprobs=None,
                        message=ChatCompletionMessage(
                            content="last message",
                            role="assistant",
                            function_call=None,
                            tool_calls=None,
                        ),
                    )
                ],
                created=1702685747,
                model="gpt-3.5-turbo-1106",
                object="chat.completion",
                system_fingerprint="fp_772e8125bb",
                usage=CompletionUsage(
                    completion_tokens=35, prompt_tokens=27, total_tokens=62
                ),
            ),
        )

    ### Tests for actionweaver.llms.openai.tools.chat.OpenAIChatCompletion

    @patch("openai.resources.chat.Completions.create")
    def test_create_message(self, mock_create):
        # Create an instance of OpenAIChatCompletion
        chat_completion = OpenAIChatCompletion(model="test")

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
        self.assertEqual(
            response,
            ChatCompletion(
                id="chatcmpl-8WCdHNVdrYkU8cir7xYcji02Lenuw",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        logprobs=None,
                        message=ChatCompletionMessage(
                            content="Hello! what can I do for you",
                            role="assistant",
                            function_call=None,
                            tool_calls=None,
                        ),
                    )
                ],
                created=1702685747,
                model="gpt-3.5-turbo-1106",
                object="chat.completion",
                system_fingerprint="fp_772e8125bb",
                usage=CompletionUsage(
                    completion_tokens=35, prompt_tokens=27, total_tokens=62
                ),
            ),
        )

    @patch("openai.resources.chat.Completions.create")
    def test_create_with_single_function(self, mock_create):
        def mock_method(text: str):
            """mock method"""
            return text

        actions = [
            Action("action1", mock_method).build_pydantic_model_cls(),
        ]
        chat_completion = OpenAIChatCompletion(model="test")

        # Define the expected functions arguments and return values in the API call
        expected_functions_and_results = [
            (
                {"tools": ["action1"], "tool_choice": "auto"},
                self.generate_single_mock_function_call_response(
                    "action1", '{\n  "text": "echo1"\n}'
                ),
            ),
            (
                {"tools": ["action1"], "tool_choice": "auto"},
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
            if "tools" in actual_call.kwargs:
                self.assertEqual(
                    [tool["function"]["name"] for tool in actual_call.kwargs["tools"]],
                    expected_functions_and_results[i][0]["tools"],
                )
                self.assertEqual(
                    actual_call.kwargs["tool_choice"],
                    expected_functions_and_results[i][0]["tool_choice"],
                )
            else:
                self.assertFalse("tools" in expected_functions_and_results[i][0])
                self.assertFalse("tool_choice" in expected_functions_and_results[i][0])

        self.assertEqual(
            messages,
            [
                {"content": "Hi!", "role": "user"},
                ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_70TwJYHkDSs80tcnF5lt8TQR",
                            function=Function(
                                arguments='{\n  "text": "echo1"\n}', name="action1"
                            ),
                            type="function",
                        )
                    ],
                ),
                {
                    "content": "echo1",
                    "name": "action1",
                    "role": "tool",
                    "tool_call_id": "call_70TwJYHkDSs80tcnF5lt8TQR",
                },
            ],
        )

        self.assertEqual(
            response,
            ChatCompletion(
                id="chatcmpl-8WCdHNVdrYkU8cir7xYcji02Lenuw",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        logprobs=None,
                        message=ChatCompletionMessage(
                            content="last message",
                            role="assistant",
                            function_call=None,
                            tool_calls=None,
                        ),
                    )
                ],
                created=1702685747,
                model="gpt-3.5-turbo-1106",
                object="chat.completion",
                system_fingerprint="fp_772e8125bb",
                usage=CompletionUsage(
                    completion_tokens=35, prompt_tokens=27, total_tokens=62
                ),
            ),
        )

    @patch("openai.resources.chat.Completions.create")
    def test_create_with_single_function_orchestration(self, mock_create):
        def mock_method(text: str):
            """mock method"""
            return text

        actions = [
            Action("action1", mock_method).build_pydantic_model_cls(),
            Action("action2", mock_method).build_pydantic_model_cls(),
        ]
        chat_completion = OpenAIChatCompletion(model="test")

        # Define the expected functions arguments and return values in the API call
        expected_functions_and_results = [
            (
                {"tools": ["action1"], "tool_choice": "auto"},
                self.generate_single_mock_function_call_response(
                    "action1", '{\n  "text": "echo1"\n}'
                ),
            ),
            (
                {
                    "tools": ["action2"],
                    "tool_choice": {
                        "function": {"name": "action2"},
                        "type": "function",
                    },
                },
                self.generate_single_mock_function_call_response(
                    "action2", '{\n  "text": "echo2"\n}'
                ),
            ),
            (
                {"tools": ["action1"], "tool_choice": "auto"},
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
            messages=messages, actions=[actions[0]], orch={actions[0]: actions[1]}
        )

        # Then
        # Use a loop to iterate over expected calls and assert function arguments in the API call
        for i, actual_call in enumerate(mock_create.call_args_list):
            if "tools" in actual_call.kwargs:
                self.assertEqual(
                    [tool["function"]["name"] for tool in actual_call.kwargs["tools"]],
                    expected_functions_and_results[i][0]["tools"],
                )
                self.assertEqual(
                    actual_call.kwargs["tool_choice"],
                    expected_functions_and_results[i][0]["tool_choice"],
                )
            else:
                self.assertFalse("tools" in expected_functions_and_results[i][0])
                self.assertFalse("tool_choice" in expected_functions_and_results[i][0])

        self.assertEqual(
            messages,
            [
                {"content": "Hi!", "role": "user"},
                ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_70TwJYHkDSs80tcnF5lt8TQR",
                            function=Function(
                                arguments='{\n  "text": "echo1"\n}', name="action1"
                            ),
                            type="function",
                        )
                    ],
                ),
                {
                    "content": "echo1",
                    "name": "action1",
                    "role": "tool",
                    "tool_call_id": "call_70TwJYHkDSs80tcnF5lt8TQR",
                },
                ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_70TwJYHkDSs80tcnF5lt8TQR",
                            function=Function(
                                arguments='{\n  "text": "echo2"\n}', name="action2"
                            ),
                            type="function",
                        )
                    ],
                ),
                {
                    "content": "echo2",
                    "name": "action2",
                    "role": "tool",
                    "tool_call_id": "call_70TwJYHkDSs80tcnF5lt8TQR",
                },
            ],
        )
        self.assertEqual(
            response,
            ChatCompletion(
                id="chatcmpl-8WCdHNVdrYkU8cir7xYcji02Lenuw",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        logprobs=None,
                        message=ChatCompletionMessage(
                            content="last message",
                            role="assistant",
                            function_call=None,
                            tool_calls=None,
                        ),
                    )
                ],
                created=1702685747,
                model="gpt-3.5-turbo-1106",
                object="chat.completion",
                system_fingerprint="fp_772e8125bb",
                usage=CompletionUsage(
                    completion_tokens=35, prompt_tokens=27, total_tokens=62
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
