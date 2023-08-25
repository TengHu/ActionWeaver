from __future__ import annotations

import unittest

from pydantic import BaseModel, create_model

from actionweaver.action import action
from actionweaver.mixins import ActionHandlerMixin


class ActionTestCase(unittest.TestCase):
    def test_action_handler_mixin(self):
        class Foo1(ActionHandlerMixin):
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


if __name__ == "__main__":
    unittest.main()
