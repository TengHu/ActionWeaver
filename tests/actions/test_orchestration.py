from __future__ import annotations

import unittest

from pydantic import BaseModel, create_model

from actionweaver.actions import Action, ActionHandlers, action
from actionweaver.actions.orchestration import (
    RequireNext,
    SelectOne,
    _ActionHandlerLLMInvoke,
    _ActionHandlerRequired,
    _ActionHandlerSelectOne,
)


class OrchestrationTestCase(unittest.TestCase):
    def mock_method():
        """mock method"""
        pass

    def test_orchestration1(self):
        # Todo: use kw-only arguments for Action
        actions = [
            Action("action1", self.mock_method, "global"),
            Action("action2", self.mock_method, "global"),
            Action("action3", self.mock_method, "global"),
            Action("action4", self.mock_method, "local1"),
            Action("action5", self.mock_method, "local2"),
        ]

        action_handler = ActionHandlers()
        for action in actions:
            action_handler.name_to_action[action.name] = action

        instance_action_handler = action_handler.bind(None).build_orchestration_dict()

        self.assertEqual(
            instance_action_handler.orch_dict,
            {
                _ActionHandlerLLMInvoke(scope="global"): _ActionHandlerSelectOne(
                    [
                        "action1",
                        "action2",
                        "action3",
                    ]
                ),
                _ActionHandlerLLMInvoke(scope="local1"): _ActionHandlerSelectOne(
                    ["action4"]
                ),
                _ActionHandlerLLMInvoke(scope="local2"): _ActionHandlerSelectOne(
                    ["action5"]
                ),
            },
        )

    def test_orchestration2(self):
        actions = [
            Action(
                "action1",
                self.mock_method,
                orchestration_expr=SelectOne(["action1", "action2"]),
            ),
            Action(
                "action2",
                self.mock_method,
            ),
            Action(
                "action3",
                self.mock_method,
                orchestration_expr=SelectOne(["action3", "action1", "action2"]),
            ),
        ]

        action_handler = ActionHandlers()
        for action in actions:
            action_handler.name_to_action[action.name] = action

        instance_action_handler = action_handler.bind(None).build_orchestration_dict()

        self.assertEqual(
            instance_action_handler.orch_dict,
            {
                _ActionHandlerLLMInvoke(scope="global"): _ActionHandlerSelectOne(
                    ["action1", "action2", "action3"]
                ),
                "action1": _ActionHandlerSelectOne(["action2"]),
                "action3": _ActionHandlerSelectOne(["action1", "action2"]),
            },
        )

    def test_orchestration3(self):
        actions = [
            Action(
                "action1",
                self.mock_method,
                orchestration_expr=SelectOne(
                    [
                        "action1",
                        SelectOne(
                            ["action3", RequireNext(["action9", "action10"]), "action4"]
                        ),
                        RequireNext(
                            ["action5", "action6", SelectOne(["action7", "action8"])]
                        ),
                    ]
                ),
            ),
        ]

        action_handler = ActionHandlers()
        for action in actions:
            action_handler.name_to_action[action.name] = action

        instance_action_handler = action_handler.bind(None).build_orchestration_dict()

        self.assertEqual(
            instance_action_handler.orch_dict,
            {
                _ActionHandlerLLMInvoke(scope="global"): ["action1"],
                "action1": _ActionHandlerSelectOne(["action3", "action5"]),
                "action3": _ActionHandlerSelectOne(["action9", "action4"]),
                "action5": _ActionHandlerRequired(action="action6"),
                "action6": _ActionHandlerRequired(action="action7"),
                "action7": _ActionHandlerSelectOne(["action8"]),
                "action9": _ActionHandlerRequired(action="action10"),
            },
        )


if __name__ == "__main__":
    unittest.main()
