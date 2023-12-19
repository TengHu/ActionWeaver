from __future__ import annotations

import unittest

from actionweaver import action
from actionweaver.actions.factories.combine import combine


class TestCase(unittest.TestCase):
    def test(self):
        @action(name="Func1")
        def func(a: int):
            """func docstring"""
            return a

        @action(name="Func2")
        def bar(b: str):
            """bar docstring"""
            return b

        combined_action = combine([func, bar], description="docstring")
        self.assertEqual(combined_action.user_method.__name__, "combine_func1_func2")
        self.assertEqual(combined_action.name, "Combine_Func1_Func2")
        self.assertEqual(combined_action.description, "docstring")

        self.assertEqual(
            combined_action(**{"Func1": {"a": 1}, "Func2": {"b": "2"}}), "1\n2"
        )


if __name__ == "__main__":
    unittest.main()
