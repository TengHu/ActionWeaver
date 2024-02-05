from __future__ import annotations

import functools
import unittest
from unittest.mock import MagicMock

from openai import AzureOpenAI, OpenAI

from actionweaver.actions import Action
from actionweaver.actions.factories.function import action

# TODO: test `enforce`` argument


class TestAction(unittest.TestCase):
    def test_action_invoke_patched_openai_client(self):
        client = OpenAI()
        mock_create = MagicMock()
        client.chat.completions.create = mock_create

        def mock_method(text: str):
            """mock method"""
            return text

        actions = [action("action1")(mock_method)]

        actions[0].invoke(
            client,
            messages=[{"role": "user", "content": "Hi!"}],
            model="test",
            stream=True,
            force=False,
        )

        mock_create.assert_called_once()

        self.assertTrue(len(mock_create.call_args_list[0].kwargs["actions"]) == 1)
        self.assertTrue(mock_create.call_args_list[0].kwargs["stream"])

        self.assertEqual(
            mock_create.call_args_list[0].kwargs["actions"][0].name, "action1"
        )

    def test_action_with_decorators_method1(self):
        def add_one(func):
            def wrapper(another_num: int):
                """Add one to the result of the function"""
                return func(another_num) + 1

            return wrapper

        @action(name="Func1")
        @add_one
        def mock_method(num: int):
            """mock method"""
            return num

        assert mock_method(1) == 2

        # This test verify that the decorator is incorporated into the pydantic model
        self.assertEqual(
            mock_method.json_schema(),
            {
                "properties": {
                    "another_num": {"title": "Another Num", "type": "integer"}
                },
                "required": ["another_num"],
                "title": "Wrapper",
                "type": "object",
            },
        )

    def test_action_with_decorators_method2(self):
        def add_one(func):
            @functools.wraps(func)
            def wrapper(another_num: int):
                return func(another_num) + 1

            return wrapper

        @action(name="Func1", decorators=[add_one])
        def mock_method(num: int):
            """mock method"""
            return num

        assert mock_method(1) == 2

        # This test verify that the decorator is not incorporated into the pydantic model
        self.assertEqual(
            mock_method.json_schema(),
            {
                "properties": {"num": {"title": "Num", "type": "integer"}},
                "required": ["num"],
                "title": "Mock_Method",
                "type": "object",
            },
        )
