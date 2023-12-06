from __future__ import annotations

import unittest

from actionweaver.actions.action import Action, ActionHandlers
from actionweaver.actions.orchestration_expr import (
    _ActionHandlerLLMInvoke,
    _ActionHandlerRequired,
    _ActionHandlerSelectOne,
)
from actionweaver.llms.openai.functions.functions import Functions


class FunctionsTestCase(unittest.TestCase):
    def mock_method(a: int, b: int, c: str = "qux"):
        """mock method"""
        pass

    def test_functions(self):
        actions = [
            Action("action1", self.mock_method, "global").build_pydantic_model_cls(),
            Action("action2", self.mock_method, "global").build_pydantic_model_cls(),
            Action("action3", self.mock_method, "global").build_pydantic_model_cls(),
        ]

        action_handler = ActionHandlers()
        for action in actions:
            action_handler.name_to_action[action.name] = action
        self.assertEqual(
            Functions.from_expr(
                _ActionHandlerSelectOne(["action1", "action2", "action3"]),
                action_handler,
            ).to_arguments(),
            {
                "functions": [
                    {
                        "description": "mock method",
                        "name": "action1",
                        "parameters": {
                            "properties": {
                                "a": {"title": "A", "type": "integer"},
                                "b": {"title": "B", "type": "integer"},
                                "c": {"default": "qux", "title": "C", "type": "string"},
                            },
                            "required": ["a", "b"],
                            "title": "Mock_Method",
                            "type": "object",
                        },
                    },
                    {
                        "description": "mock method",
                        "name": "action2",
                        "parameters": {
                            "properties": {
                                "a": {"title": "A", "type": "integer"},
                                "b": {"title": "B", "type": "integer"},
                                "c": {"default": "qux", "title": "C", "type": "string"},
                            },
                            "required": ["a", "b"],
                            "title": "Mock_Method",
                            "type": "object",
                        },
                    },
                    {
                        "description": "mock method",
                        "name": "action3",
                        "parameters": {
                            "properties": {
                                "a": {"title": "A", "type": "integer"},
                                "b": {"title": "B", "type": "integer"},
                                "c": {"default": "qux", "title": "C", "type": "string"},
                            },
                            "required": ["a", "b"],
                            "title": "Mock_Method",
                            "type": "object",
                        },
                    },
                ],
                "function_call": "auto",
            },
        )

        self.assertEqual(
            Functions.from_expr(
                _ActionHandlerRequired("action1"),
                action_handler,
            ).to_arguments(),
            {
                "functions": [
                    {
                        "description": "mock method",
                        "name": "action1",
                        "parameters": {
                            "properties": {
                                "a": {"title": "A", "type": "integer"},
                                "b": {"title": "B", "type": "integer"},
                                "c": {"default": "qux", "title": "C", "type": "string"},
                            },
                            "required": ["a", "b"],
                            "title": "Mock_Method",
                            "type": "object",
                        },
                    }
                ],
                "function_call": {"name": "action1"},
            },
        )

        self.assertEqual(
            Functions.from_expr(
                _ActionHandlerLLMInvoke("global"),
                action_handler,
            ).to_arguments(),
            {
                "function_call": None,
                "functions": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
