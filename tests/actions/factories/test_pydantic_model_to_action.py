from __future__ import annotations

import unittest
from typing import List

from pydantic import BaseModel

from actionweaver.actions.factories.pydantic_model_to_action import from_model


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
        action = from_model(
            TestCase.Place, name="Func1", description="Extract Place model"
        )
        self.assertTrue("create_place" in action.decorated_method.__name__)
        self.assertEqual(action.name, "Func1")
        self.assertEqual(action.stop, True)
        self.assertEqual(action.description, "Extract Place model")
        self.assertEqual(
            action(**{"place": {"lat": 23, "lng": 23, "description": "description"}}),
            TestCase.Place(lat=23.0, lng=23.0, description="description"),
        )


        action = from_model(
            TestCase.Places, name="Func2", description="Extract Places model"
        )

        self.assertTrue("create_place" in action.decorated_method.__name__)

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


if __name__ == "__main__":
    unittest.main()
