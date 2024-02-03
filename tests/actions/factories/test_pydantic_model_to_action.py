from __future__ import annotations

import functools
import unittest
from typing import List

from pydantic import BaseModel

from actionweaver.actions.factories.pydantic_model_to_action import action_from_model


class TestCase(unittest.TestCase):
    class Place(BaseModel):
        """A geo location"""

        lat: float
        lng: float
        description: str

    class Places(BaseModel):
        """A geo location"""

        places: List[TestCase.Place]

    def test_from_pydantic_model(self):
        action = action_from_model(
            TestCase.Place, name="Func1", description="Extract Place model"
        )
        self.assertTrue("create_place" in action.function.__name__)
        self.assertEqual(action.name, "Func1")
        self.assertEqual(action.stop, True)
        self.assertEqual(action.description, "Extract Place model")
        self.assertEqual(
            action(**{"place": {"lat": 23, "lng": 23, "description": "description"}}),
            TestCase.Place(lat=23.0, lng=23.0, description="description"),
        )

        action = action_from_model(
            TestCase.Places, name="Func2", description="Extract Places model"
        )

        self.assertTrue("create_place" in action.function.__name__)

        self.assertEqual(action.name, "Func2")
        self.assertEqual(action.stop, True)
        self.assertEqual(action.description, "Extract Places model")

        self.assertEqual(
            action(
                **{
                    "places": {
                        "places": [{"lat": 23, "lng": 23, "description": "description"}]
                    }
                }
            ),
            TestCase.Places(
                places=[TestCase.Place(lat=23.0, lng=23.0, description="description")]
            ),
        )

    def test_from_pydantic_model_with_decorators(self):
        def decorator(func):
            @functools.wraps(func)
            def duplicate(*args, **kwargs):
                result = func(*args, **kwargs)
                return result, result

            return duplicate

        action = action_from_model(
            TestCase.Place, name="Func1", description="Extract Place model"
        )
        self.assertTrue("create_place" in action.function.__name__)
        self.assertEqual(action.name, "Func1")
        self.assertEqual(action.stop, True)
        self.assertEqual(action.description, "Extract Place model")
        self.assertEqual(
            action(**{"place": {"lat": 23, "lng": 23, "description": "description"}}),
            TestCase.Place(lat=23.0, lng=23.0, description="description"),
        )

        action = action_from_model(
            TestCase.Places,
            name="Func2",
            description="Extract Places model",
            decorators=[decorator],
        )

        self.assertTrue("create_place" in action.function.__name__)
        self.assertEqual(len(action.decorators), 1)

        self.assertEqual(action.name, "Func2")
        self.assertEqual(action.stop, True)
        self.assertEqual(action.description, "Extract Places model")

        actual = action(
            **{
                "places": {
                    "places": [{"lat": 23, "lng": 23, "description": "description"}]
                }
            }
        )

        # This test verify that the decorator is not incorporated into the pydantic model
        self.assertTrue(
            action.json_schema()["properties"], {"places": {"$ref": "#/$defs/Places"}}
        )
        self.assertEqual(
            action.json_schema()["$defs"],
            {
                "Place": {
                    "description": "A geo location",
                    "properties": {
                        "description": {"title": "Description", "type": "string"},
                        "lat": {"title": "Lat", "type": "number"},
                        "lng": {"title": "Lng", "type": "number"},
                    },
                    "required": ["lat", "lng", "description"],
                    "title": "Place",
                    "type": "object",
                },
                "Places": {
                    "description": "A geo location",
                    "properties": {
                        "places": {
                            "items": {"$ref": "#/$defs/Place"},
                            "title": "Places",
                            "type": "array",
                        }
                    },
                    "required": ["places"],
                    "title": "Places",
                    "type": "object",
                },
            },
        )
        self.assertEqual(len(actual), 2)


if __name__ == "__main__":
    unittest.main()
