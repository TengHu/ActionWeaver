from __future__ import annotations

import unittest
from unittest.mock import Mock, call, patch
from urllib import response

from actionweaver.actions import Action
from actionweaver.llms.general.action_processor import ActionProcessor


class TestActionProcessor(unittest.TestCase):
    def test_action_processor(self):
        def get_current_weather(location, unit="fahrenheit"):
            """mock method"""
            import json

            return json.dumps(
                {"location": location, "temperature": "22", "unit": "celsius"}
            )

        ap = ActionProcessor(
            tools=[
                Action(
                    "GetWeather",
                    get_current_weather,
                ).build_pydantic_model_cls()
            ]
        )

        response, ok, err = ap.respond("hello")
        self.assertFalse(ok)
        self.assertTrue(response is None)
        self.assertTrue(
            "Unable to extract a valid function from the input. Error encountered in extractor"
            in err,
        )

        response, ok, err = ap.respond(
            '{\n  "function": "GetWeather",\n  "parameters": {\n    "location": "San Francisco",\n    "unit": "fahrenheit"\n  }\n}'
        )
        self.assertTrue(ok)
        self.assertTrue(
            response,
            {
                "location": "San Francisco",
                "temperature": "22",
                "unit": "celsius",
            },
        )
        self.assertTrue(err is None)

    def test_action_processor_with_custom_extractor(self):
        def get_current_weather(location, unit="fahrenheit"):
            """mock method"""
            import json

            return json.dumps(
                {"location": location, "temperature": "22", "unit": "celsius"}
            )

        def extractor(text: str):
            import json

            j = json.loads(text)
            return {"name": j["tool_name"], "parameters": j["tool_arguments"]}

        ap = ActionProcessor(
            tools=[
                Action(
                    "GetWeather",
                    get_current_weather,
                ).build_pydantic_model_cls()
            ],
            custom_extractor=extractor,
        )

        response, ok, err = ap.respond("hello")
        self.assertFalse(ok)
        self.assertTrue(response is None)
        self.assertTrue(
            "Unable to extract a valid function from the input. Error encountered in extractor"
            in err,
        )

        response, ok, err = ap.respond(
            '{\n  "tool_name": "GetWeather",\n  "tool_arguments": {\n    "location": "San Francisco",\n    "unit": "fahrenheit"\n  }\n}'
        )
        self.assertTrue(ok)
        self.assertTrue(
            response,
            {
                "location": "San Francisco",
                "temperature": "22",
                "unit": "celsius",
            },
        )
        self.assertTrue(err is None)


if __name__ == "__main__":
    unittest.main()
