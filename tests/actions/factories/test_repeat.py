from __future__ import annotations

import unittest

from actionweaver import action
from actionweaver.actions.factories.repeat import repeat


class TestCase(unittest.TestCase):
    def test(self):
        @action(name="Func1")
        def func(a: int):
            """docstring"""
            return a

        repeated_action = repeat(func)
        self.assertEqual(repeated_action.decorated_method.__name__, "func")
        self.assertEqual(repeated_action.name, "Func1")
        self.assertEqual(repeated_action.description, "docstring")
        self.assertEqual(repeated_action(**{"Func1": [{"a": 1}, {"a": 2}]}), "1\n2")


if __name__ == "__main__":
    unittest.main()
