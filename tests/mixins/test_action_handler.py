from __future__ import annotations

import unittest

from actionweaver.actions import action
from actionweaver.actions.action import ActionException
from actionweaver.actions.orchestration_expr import RequireNext, SelectOne
from actionweaver.llms.openai.chat import OpenAIChatCompletion
from actionweaver.mixins import ActionHandlerMixin


class ActionTestCase(unittest.TestCase):
    def test_action_handler_mixin(self):
        class Foo1(ActionHandlerMixin):
            def __init__(self) -> None:
                super().__init__()
                self.llm = OpenAIChatCompletion(model="model")

            @action("Sum")
            def sum_bar(self, bar1: int, bar2: int):
                """
                Sum bar1 and bar2.
                """
                return bar1 + bar2

        class Foo2(Foo1):
            @action("Minus")
            def minus_func(self, bar1: int, bar2: int):
                """
                Minus bar1 and bar2.
                """
                return bar1 - bar2

        foo1 = Foo1()
        foo2 = Foo2()

        self.assertEqual(foo1.sum_bar(1, 2), 3)
        self.assertEqual(len(foo1._action_handlers), 1)

        self.assertEqual(foo2.sum_bar(1, 2), 3)
        self.assertEqual(foo2.minus_func(1, 2), -1)
        self.assertEqual(len(foo2._action_handlers), 2)

    def test_action_handler_mixin_with_invalid_orchestration(self):
        try:

            class Foo(ActionHandlerMixin):
                @action(
                    "Sum",
                    orch_expr=SelectOne(
                        ["Sum", RequireNext(["Sum", SelectOne(["Sum", "action3"])])]
                    ),
                )
                def mock_method(self, bar1: int, bar2: int):
                    """mock method"""
                    pass

        except ActionException as e:
            self.assertEqual(
                str(e),
                "Action action3 not found.",
            )


if __name__ == "__main__":
    unittest.main()
