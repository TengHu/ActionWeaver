import functools
import unittest
from typing import Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag, validate_call
from typing_extensions import Annotated

from actionweaver.utils.cache import cache, lru_cache, preserve_original_signature
from actionweaver.utils.pydantic_utils import create_pydantic_model_from_func


class UtilsTestCase(unittest.TestCase):
    def test_create_pydantic_model_from_func1(self):
        """Test create_pydantic_model_from_func with a simple function"""

        def foo(bar1: int, bar2: int, bar3: str = "qux"):
            pass

        Foo = create_pydantic_model_from_func("Foo", foo)
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

        Foo = create_pydantic_model_from_func("Foo", foo)
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

    def test_create_pydantic_model_from_func2(self):
        """Test create_pydantic_model_from_func with a function that has a pydantic argument"""

        class Person(BaseModel):
            first_name: str
            last_name: str
            age: int
            email: str

        class Persons(BaseModel):
            persons: list[Person]

        def bar(a: int, persons: Persons):
            pass

        def foo(a: int, person: Person):
            pass

        Foo = create_pydantic_model_from_func("Foo", foo)

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

        Bar = create_pydantic_model_from_func("Bar", bar)

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

        Foo = create_pydantic_model_from_func("Foo", foo1)
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

        Foo = create_pydantic_model_from_func("Foo", foo2)
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

        Foo = create_pydantic_model_from_func("Foo", foo3)
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
            "Foo", foo, override_params={"a": (int, ...), "person": (Person, ...)}
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
            "Bar", bar, override_params={"a": (int, ...), "persons": (Persons, ...)}
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

    def test_create_pydantic_model_from_func_with_ignored_params(self):
        def foo(a, b, c):
            pass

        def bar(a, b, c=2):
            pass

        Foo = create_pydantic_model_from_func("Foo", foo, ignored_params=["a"])
        self.assertEqual(
            Foo.model_json_schema(),
            {
                "properties": {"b": {"title": "B"}, "c": {"title": "C"}},
                "required": ["b", "c"],
                "title": "Foo",
                "type": "object",
            },
        )
        Bar = create_pydantic_model_from_func("Bar", bar, ignored_params=["b", "c"])
        self.assertEqual(
            Bar.model_json_schema(),
            {
                "properties": {"a": {"title": "A"}},
                "required": ["a"],
                "title": "Bar",
                "type": "object",
            },
        )

    def test_create_pydantic_model_from_func_with_field_annotation1(self):

        @validate_call
        def how_many(num: Annotated[int, Field(gt=10)]):
            return num

        HowMany = create_pydantic_model_from_func(
            "HowMany",
            how_many,
        )

        self.assertEqual(
            HowMany.model_json_schema(),
            {
                "properties": {
                    "num": {"exclusiveMinimum": 10, "title": "Num", "type": "integer"}
                },
                "title": "HowMany",
                "type": "object",
            },
        )

    def test_create_pydantic_model_from_func_with_field_annotation2(self):

        class Cat(BaseModel):
            pet_type: Literal["cat"]
            age: int

        class Dog(BaseModel):
            pet_kind: Literal["dog"]
            age: int

        def pet_discriminator(v):
            if isinstance(v, dict):
                return v.get("pet_type", v.get("pet_kind"))
            return getattr(v, "pet_type", getattr(v, "pet_kind", None))

        class Model(BaseModel):
            pet: Union[Annotated[Cat, Tag("cat")], Annotated[Dog, Tag("dog")]] = Field(
                discriminator=Discriminator(pet_discriminator)
            )

        def foo(model: Model):
            pass

        Foo = create_pydantic_model_from_func(
            "Foo",
            foo,
        )

        self.assertEqual(
            Foo.model_json_schema(),
            {
                "$defs": {
                    "Cat": {
                        "properties": {
                            "pet_type": {"const": "cat", "title": "Pet Type"},
                            "age": {"title": "Age", "type": "integer"},
                        },
                        "required": ["pet_type", "age"],
                        "title": "Cat",
                        "type": "object",
                    },
                    "Dog": {
                        "properties": {
                            "pet_kind": {"const": "dog", "title": "Pet Kind"},
                            "age": {"title": "Age", "type": "integer"},
                        },
                        "required": ["pet_kind", "age"],
                        "title": "Dog",
                        "type": "object",
                    },
                    "Model": {
                        "properties": {
                            "pet": {
                                "oneOf": [
                                    {"$ref": "#/$defs/Cat"},
                                    {"$ref": "#/$defs/Dog"},
                                ],
                                "title": "Pet",
                            }
                        },
                        "required": ["pet"],
                        "title": "Model",
                        "type": "object",
                    },
                },
                "properties": {"model": {"$ref": "#/$defs/Model"}},
                "required": ["model"],
                "title": "Foo",
                "type": "object",
            },
        )


if __name__ == "__main__":
    unittest.main()
