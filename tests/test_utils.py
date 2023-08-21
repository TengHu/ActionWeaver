from __future__ import annotations

import functools
import unittest

from pydantic import BaseModel, create_model

from actionweaver.cache import cache, lru_cache, preserve_original_signature
from actionweaver.utils import create_pydantic_model_from_signature


class UtilsTestCase(unittest.TestCase):
    def test_create_pydantic_model_from_signature(self):
        def foo(bar1: int, bar2: int, bar3: str = "qux"):
            pass

        Foo = create_pydantic_model_from_signature(foo, "Foo")
        self.assertEqual(
            Foo.model_json_schema(),
            {
                "properties": {
                    "bar1": {"title": "Bar1", "type": "integer"},
                    "bar2": {"title": "Bar2", "type": "integer"},
                    "bar3": {"default": "qux", "title": "Bar3", "type": "string"},
                },
                "required": ["bar1", "bar2"],
                "title": "Foo",
                "type": "object",
            },
        )

        # ignore self argument
        def foo(self, bar1: int, bar2: int, bar3: str = "qux"):
            pass

        Foo = create_pydantic_model_from_signature(foo, "Foo")
        self.assertEqual(
            Foo.model_json_schema(),
            {
                "properties": {
                    "bar1": {"title": "Bar1", "type": "integer"},
                    "bar2": {"title": "Bar2", "type": "integer"},
                    "bar3": {"default": "qux", "title": "Bar3", "type": "string"},
                },
                "required": ["bar1", "bar2"],
                "title": "Foo",
                "type": "object",
            },
        )

    def test_create_pydantic_model_from_signature_with_decorators(self):
        @cache
        def foo1(bar1: int, bar2: int, bar3: str = "qux"):
            """foo"""
            pass

        Foo = create_pydantic_model_from_signature(foo1, "Foo")
        self.assertEqual(
            Foo.model_json_schema(),
            {
                "properties": {
                    "bar1": {"title": "Bar1", "type": "integer"},
                    "bar2": {"title": "Bar2", "type": "integer"},
                    "bar3": {"default": "qux", "title": "Bar3", "type": "string"},
                },
                "required": ["bar1", "bar2"],
                "title": "Foo",
                "type": "object",
            },
        )

        @lru_cache(128)
        def foo2(bar1: int, bar2: int, bar3: str = "qux"):
            """foo"""
            pass

        Foo = create_pydantic_model_from_signature(foo2, "Foo")
        self.assertEqual(
            Foo.model_json_schema(),
            {
                "properties": {
                    "bar1": {"title": "Bar1", "type": "integer"},
                    "bar2": {"title": "Bar2", "type": "integer"},
                    "bar3": {"default": "qux", "title": "Bar3", "type": "string"},
                },
                "required": ["bar1", "bar2"],
                "title": "Foo",
                "type": "object",
            },
        )

        @preserve_original_signature
        def custom_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            import inspect

            # Todo: must have this, verify this
            wrapper.__signature__ = inspect.signature(func)
            return wrapper

        @custom_decorator
        def foo3(bar1: int, bar2: int, bar3: str = "qux"):
            """foo"""
            pass

        Foo = create_pydantic_model_from_signature(foo3, "Foo")
        self.assertEqual(
            Foo.model_json_schema(),
            {
                "properties": {
                    "bar1": {"title": "Bar1", "type": "integer"},
                    "bar2": {"title": "Bar2", "type": "integer"},
                    "bar3": {"default": "qux", "title": "Bar3", "type": "string"},
                },
                "required": ["bar1", "bar2"],
                "title": "Foo",
                "type": "object",
            },
        )


if __name__ == "__main__":
    unittest.main()
