from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from openai import AzureOpenAI, OpenAI

from actionweaver.actions import Action

# TODO: test `enforce`` argument


class TestAction(unittest.TestCase):
    def test_action_invoke_patched_openai(self):
        client = OpenAI()
        mock_create = MagicMock()
        client.chat.completions.create = mock_create

        def mock_method(text: str):
            """mock method"""
            return text

        actions = [
            Action("action1", mock_method).build_pydantic_model_cls(),
        ]

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

    def test_action_invoke_patched_azure_openai(self):
        client = AzureOpenAI(
            azure_endpoint="AZURE_OPENAI_ENDPOINT",
            api_key="AZURE_OPENAI_KEY",
            api_version="2023-10-01-preview",
        )
        mock_create = MagicMock()
        client.chat.completions.create = mock_create

        def mock_method(text: str):
            """mock method"""
            return text

        actions = [
            Action("action1", mock_method).build_pydantic_model_cls(),
        ]

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
