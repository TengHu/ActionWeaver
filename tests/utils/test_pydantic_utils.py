from __future__ import annotations

import functools
import unittest

from pydantic import BaseModel, create_model

from actionweaver.utils.cache import cache, lru_cache, preserve_original_signature
from actionweaver.utils.pydantic_utils import create_pydantic_model_from_func


class UtilsTestCase(unittest.TestCase):
    def test_create_pydantic_model_from_func(self):
        def foo(bar1: int, bar2: int, bar3: str = "qux"):
            pass

        Foo = create_pydantic_model_from_func(foo, "Foo")
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

        Foo = create_pydantic_model_from_func(foo, "Foo")
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

    def test_create_pydantic_model_from_func_with_pydantic_argument(self):
        class Person(BaseModel):
            first_name: str
            last_name: str
            age: int
            email: str

        class Persons(BaseModel):
            persons: list[Person]

        def foo(a: int, person: Person):
            pass

        def bar(a: int, persons: Persons):
            pass

        Foo = create_pydantic_model_from_func(foo, "Foo", models=[Person])

        self.assertEqual(
            Foo.model_json_schema(),
            {
                "$defs": {
                    "Person": {
                        "properties": {
                            "first_name": {"title": "First Name", "type": "string"},
                            "last_name": {"title": "Last Name", "type": "string"},
                            "age": {"title": "Age", "type": "integer"},
                            "email": {"title": "Email", "type": "string"},
                        },
                        "required": ["first_name", "last_name", "age", "email"],
                        "title": "Person",
                        "type": "object",
                    }
                },
                "properties": {
                    "a": {"title": "A", "type": "integer"},
                    "person": {"$ref": "#/$defs/Person"},
                },
                "required": ["a", "person"],
                "title": "Foo",
                "type": "object",
            },
        )

        Bar = create_pydantic_model_from_func(bar, "Bar", models=[Persons])

        self.assertEqual(
            Bar.model_json_schema(),
            {
                "$defs": {
                    "Person": {
                        "properties": {
                            "first_name": {"title": "First Name", "type": "string"},
                            "last_name": {"title": "Last Name", "type": "string"},
                            "age": {"title": "Age", "type": "integer"},
                            "email": {"title": "Email", "type": "string"},
                        },
                        "required": ["first_name", "last_name", "age", "email"],
                        "title": "Person",
                        "type": "object",
                    },
                    "Persons": {
                        "properties": {
                            "persons": {
                                "items": {"$ref": "#/$defs/Person"},
                                "title": "Persons",
                                "type": "array",
                            }
                        },
                        "required": ["persons"],
                        "title": "Persons",
                        "type": "object",
                    },
                },
                "properties": {
                    "a": {"title": "A", "type": "integer"},
                    "persons": {"$ref": "#/$defs/Persons"},
                },
                "required": ["a", "persons"],
                "title": "Bar",
                "type": "object",
            },
        )

    def test_create_pydantic_model_from_func_with_decorators(self):
        @cache
        def foo1(bar1: int, bar2: int, bar3: str = "qux"):
            """foo"""
            pass

        Foo = create_pydantic_model_from_func(foo1, "Foo")
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

        Foo = create_pydantic_model_from_func(foo2, "Foo")
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

        Foo = create_pydantic_model_from_func(foo3, "Foo")
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

    def test_create_pydantic_model_from_func_with_override_params(self):
        class Person(BaseModel):
            first_name: str
            last_name: str
            age: int
            email: str

        class Persons(BaseModel):
            persons: list[Person]

        def foo():
            pass

        def bar():
            pass

        Foo = create_pydantic_model_from_func(
            foo, "Foo", override_params={"a": (int, ...), "person": (Person, ...)}
        )

        self.assertEqual(
            Foo.model_json_schema(),
            {
                "$defs": {
                    "Person": {
                        "properties": {
                            "first_name": {"title": "First Name", "type": "string"},
                            "last_name": {"title": "Last Name", "type": "string"},
                            "age": {"title": "Age", "type": "integer"},
                            "email": {"title": "Email", "type": "string"},
                        },
                        "required": ["first_name", "last_name", "age", "email"],
                        "title": "Person",
                        "type": "object",
                    }
                },
                "properties": {
                    "a": {"title": "A", "type": "integer"},
                    "person": {"$ref": "#/$defs/Person"},
                },
                "required": ["a", "person"],
                "title": "Foo",
                "type": "object",
            },
        )

        Bar = create_pydantic_model_from_func(
            bar, "Bar", override_params={"a": (int, ...), "persons": (Persons, ...)}
        )

        self.assertEqual(
            Bar.model_json_schema(),
            {
                "$defs": {
                    "Person": {
                        "properties": {
                            "first_name": {"title": "First Name", "type": "string"},
                            "last_name": {"title": "Last Name", "type": "string"},
                            "age": {"title": "Age", "type": "integer"},
                            "email": {"title": "Email", "type": "string"},
                        },
                        "required": ["first_name", "last_name", "age", "email"],
                        "title": "Person",
                        "type": "object",
                    },
                    "Persons": {
                        "properties": {
                            "persons": {
                                "items": {"$ref": "#/$defs/Person"},
                                "title": "Persons",
                                "type": "array",
                            }
                        },
                        "required": ["persons"],
                        "title": "Persons",
                        "type": "object",
                    },
                },
                "properties": {
                    "a": {"title": "A", "type": "integer"},
                    "persons": {"$ref": "#/$defs/Persons"},
                },
                "required": ["a", "persons"],
                "title": "Bar",
                "type": "object",
            },
        )


if __name__ == "__main__":
    unittest.main()
