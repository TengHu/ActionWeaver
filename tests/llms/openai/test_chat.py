from __future__ import annotations

import unittest
from unittest.mock import Mock, call, patch

from openai.openai_object import OpenAIObject

from actionweaver.actions import Action, ActionHandlers, action
from actionweaver.actions.orchestration import RequireNext, SelectOne
from actionweaver.llms.openai.chat import OpenAIChatCompletion


class TestOpenAIChatCompletion(unittest.TestCase):
    def generate_mock_function_call_response(self, name, arguments):
        mock_response = OpenAIObject()
        mock_response.choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "function_call": OpenAIObject.construct_from(
                        {
                            "name": name,
                            "arguments": arguments,
                        }
                    ),
                },
                "finish_reason": "function_call",
            }
        ]
        mock_response.usage = {
            "prompt_tokens": 1413,
            "completion_tokens": 118,
            "total_tokens": 1531,
        }
        return mock_response

    def generate_mock_message_response(self, content):
        mock_response = OpenAIObject()
        mock_response.choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
        mock_response.usage = {
            "prompt_tokens": 1413,
            "completion_tokens": 118,
            "total_tokens": 1531,
        }
        return mock_response

    @patch("openai.ChatCompletion.create")
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
                {"role": "assistant", "content": "Hello! what can I do for you"},
            ],
        )
        self.assertEqual(response, "Hello! what can I do for you")

    @patch("openai.ChatCompletion.create")
    def test_create_with_functions1(self, mock_create):
        def mock_method(self, text: str):
            """mock method"""
            return text

        # Create an instance of OpenAIChatCompletion with action handlers
        actions = [
            Action("action1", mock_method).build_pydantic_model_cls(),
        ]
        action_handler = ActionHandlers()
        action_handler.name_to_action["action1"] = actions[0]
        instance_action_handler = action_handler.bind(None).build_orchestration_dict()
        chat_completion = OpenAIChatCompletion(model="test")
        chat_completion._bind_action_handlers(instance_action_handler)

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
        response = chat_completion.create(messages=messages)

        # Then
        # Use a loop to iterate over expected calls and assert function arguments in the API call
        for i, actual_call in enumerate(mock_create.call_args_list):
            self.assertEqual(
                [func["name"] for func in actual_call.kwargs["functions"]],
                expected_functions_and_results[i][0]["functions"],
            )
            self.assertEqual(
                actual_call.kwargs["function_call"],
                expected_functions_and_results[i][0]["function_call"],
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
                {"content": "last message", "role": "assistant"},
            ],
        )
        self.assertEqual(response, "last message")

    @patch("openai.ChatCompletion.create")
    def test_create_with_functions2(self, mock_create):
        def mock_method(self, text: str):
            """mock method"""
            return text

        # Create an instance of OpenAIChatCompletion with action handlers
        actions = [
            Action(
                "action1",
                mock_method,
                orch_expr=SelectOne(["action1", "action2", "action3"]),
            ).build_pydantic_model_cls(),
            Action(
                "action2",
                mock_method,
                orch_expr=RequireNext(["action2", "action3"]),
            ).build_pydantic_model_cls(),
            Action(
                "action3",
                mock_method,
                orch_expr=RequireNext(["action3", "action4"]),
            ).build_pydantic_model_cls(),
            Action(
                "action4",
                mock_method,
            ).build_pydantic_model_cls(),
        ]
        action_handler = ActionHandlers()
        action_handler.name_to_action["action1"] = actions[0]
        action_handler.name_to_action["action2"] = actions[1]
        action_handler.name_to_action["action3"] = actions[2]
        action_handler.name_to_action["action4"] = actions[3]
        instance_action_handler = action_handler.bind(None).build_orchestration_dict()
        chat_completion = OpenAIChatCompletion(model="test")
        chat_completion._bind_action_handlers(instance_action_handler)

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
                {
                    "functions": ["action1", "action2", "action3", "action4"],
                    "function_call": "auto",
                },
                self.generate_mock_message_response("last message"),
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
        # Use a loop to iterate over expected calls and assert function arguments in the API call
        for i, actual_call in enumerate(mock_create.call_args_list):
            self.assertEqual(
                [func["name"] for func in actual_call.kwargs["functions"]],
                expected_functions_and_results[i][0]["functions"],
            )
            self.assertEqual(
                actual_call.kwargs["function_call"],
                expected_functions_and_results[i][0]["function_call"],
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
                {"content": "last message", "role": "assistant"},
            ],
        )
        self.assertEqual(response, "last message")

    @patch("openai.ChatCompletion.create")
    def test_create_with_llm_orchestration_expr(self, mock_create):
        def mock_method(self, text: str):
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

        action_handler = ActionHandlers()
        action_handler.name_to_action["action1"] = actions[0]
        action_handler.name_to_action["action2"] = actions[1]
        action_handler.name_to_action["action3"] = actions[2]
        action_handler.name_to_action["action4"] = actions[3]
        instance_action_handler = action_handler.bind(None).build_orchestration_dict()
        chat_completion = OpenAIChatCompletion(model="test")
        chat_completion._bind_action_handlers(instance_action_handler)

        # Define the expected functions arguments and return values in the API call
        expected_functions_and_results = [
            (
                {
                    "functions": ["action1", "action2", "action3"],
                    "function_call": "auto",
                },
                self.generate_mock_function_call_response(
                    "action1", '{\n  "text": "echo1"\n}'
                ),
            ),
            (
                {
                    "functions": ["action1", "action2", "action3"],
                    "function_call": "auto",
                },
                self.generate_mock_function_call_response(
                    "action2", '{\n  "text": "echo2"\n}'
                ),
            ),
            (
                {
                    "functions": ["action1", "action2", "action3"],
                    "function_call": "auto",
                },
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
                {
                    "functions": ["action1", "action2", "action3"],
                    "function_call": "auto",
                },
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
            orch_expr=SelectOne(
                ["action1", "action2", RequireNext(["action3", "action4"])]
            ),
        )

        # Then
        # Use a loop to iterate over expected calls and assert function arguments in the API call
        for i, actual_call in enumerate(mock_create.call_args_list):
            self.assertEqual(
                [func["name"] for func in actual_call.kwargs["functions"]],
                expected_functions_and_results[i][0]["functions"],
            )
            self.assertEqual(
                actual_call.kwargs["function_call"],
                expected_functions_and_results[i][0]["function_call"],
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
                {"content": "last message", "role": "assistant"},
            ],
        )
        self.assertEqual(response, "last message")


if __name__ == "__main__":
    unittest.main()
