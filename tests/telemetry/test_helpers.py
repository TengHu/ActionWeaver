import logging
import unittest
from unittest import mock
from unittest.mock import Mock

from actionweaver.telemetry import get_parent_run_id, traceable


class TestTraceable(unittest.TestCase):
    def test_traceable1(self):
        mock_logger = Mock()

        @traceable("GetCurrentWeather", mock_logger, level=logging.INFO)
        def get_current_weather(location, unit="fahrenheit"):
            """mock method"""
            import json

            return json.dumps(
                {"location": location, "temperature": "22", "unit": "celsius"}
            )

        self.assertEqual(
            get_current_weather("Berlin"),
            '{"location": "Berlin", "temperature": "22", "unit": "celsius"}',
        )

        self.assertEqual(len(mock_logger.log.call_args_list), 1)
        self.assertEqual(mock_logger.log.call_args_list[0].args[0], logging.INFO)
        self.assertEqual(
            mock_logger.log.call_args_list[0].args[1]["name"], "GetCurrentWeather"
        )
        self.assertEqual(
            mock_logger.log.call_args_list[0].args[1]["inputs"],
            {"location": "Berlin", "unit": "fahrenheit"},
        )
        self.assertEqual(
            mock_logger.log.call_args_list[0].args[1]["outputs"],
            '{"location": "Berlin", "temperature": "22", "unit": "celsius"}',
        )
        self.assertEqual(
            mock_logger.log.call_args_list[0].args[1]["parent_run_id"], None
        )
        self.assertTrue("run_id" in mock_logger.log.call_args_list[0].args[1])

    def test_traceable_with_exception(self):
        mock_logger = Mock()

        @traceable("GetCurrentWeather", mock_logger, level=logging.INFO)
        def get_current_weather(location, unit="fahrenheit"):
            """mock method"""

            raise Exception("test exception")

        try:
            get_current_weather("Berlin"),
        except Exception:
            pass

        self.assertEqual(len(mock_logger.log.call_args_list), 1)
        self.assertEqual(mock_logger.log.call_args_list[0].args[0], logging.INFO)
        self.assertEqual(
            mock_logger.log.call_args_list[0].args[1]["name"], "GetCurrentWeather"
        )
        self.assertEqual(
            mock_logger.log.call_args_list[0].args[1]["inputs"],
            {"location": "Berlin", "unit": "fahrenheit"},
        )
        self.assertTrue(
            'raise Exception("test exception")'
            in mock_logger.log.call_args_list[0].args[1]["error"]
        )
        self.assertEqual(
            mock_logger.log.call_args_list[0].args[1]["parent_run_id"], None
        )
        self.assertTrue("run_id" in mock_logger.log.call_args_list[0].args[1])

    def test_traceable_with_nested_functions1(self):
        mock_logger = Mock()

        @traceable("MockFunction1", mock_logger, level=logging.INFO)
        def mock_method1(number: int):
            """mock method"""

            return number + 1

        @traceable("MockFunction2", mock_logger, level=logging.INFO)
        def mock_method2(number: int):
            """mock method"""

            return mock_method1(number) + 1

        self.assertEqual(mock_method2(1), 3)
        self.assertEqual(len(mock_logger.log.call_args_list), 2)

        # MockFunction1
        self.assertEqual(
            mock_logger.log.call_args_list[0].args[1]["name"], "MockFunction1"
        )
        self.assertTrue(
            mock_logger.log.call_args_list[0].args[1]["parent_run_id"] is not None
        )
        self.assertEqual(mock_logger.log.call_args_list[0].args[1]["outputs"], 2)
        self.assertTrue("run_id" in mock_logger.log.call_args_list[0].args[1])

        # MockFunction2
        self.assertEqual(
            mock_logger.log.call_args_list[1].args[1]["name"], "MockFunction2"
        )
        self.assertTrue(
            mock_logger.log.call_args_list[1].args[1]["parent_run_id"] is None
        )
        self.assertEqual(mock_logger.log.call_args_list[1].args[1]["outputs"], 3)
        self.assertTrue("run_id" in mock_logger.log.call_args_list[1].args[1])

    def test_traceable_with_nested_functions2(self):
        mock_logger = Mock()

        @traceable("MockFunction1", mock_logger, level=logging.INFO)
        def mock_method1(number: int):
            """mock method"""

            return number + 1

        @traceable("MockFunction2", mock_logger, level=logging.INFO)
        def mock_method2(number: int):
            """mock method"""
            num1 = mock_method1(number)
            num2 = mock_method1(number)

            mock_logger.info(
                {"message": "log something", "parent_run_id": get_parent_run_id()}
            )

            return num1 + num2

        @traceable("MockFunction3", mock_logger, level=logging.INFO)
        def mock_method3(number: int):
            """mock method"""

            return mock_method2(number) + 1

        self.assertEqual(mock_method3(1), 5)

        self.assertEqual(len(mock_logger.log.call_args_list), 4)

        # MockFunction1 first call
        self.assertEqual(
            mock_logger.log.call_args_list[0].args[1]["name"], "MockFunction1"
        )
        self.assertTrue(
            mock_logger.log.call_args_list[0].args[1]["parent_run_id"] is not None
        )
        mock_function1_first_parent_run_id = mock_logger.log.call_args_list[0].args[1][
            "parent_run_id"
        ]
        self.assertEqual(mock_logger.log.call_args_list[0].args[1]["outputs"], 2)
        self.assertTrue("run_id" in mock_logger.log.call_args_list[0].args[1])

        # MockFunction1 second call
        self.assertEqual(
            mock_logger.log.call_args_list[1].args[1]["name"], "MockFunction1"
        )
        self.assertTrue(
            mock_logger.log.call_args_list[1].args[1]["parent_run_id"] is not None
        )
        mock_function1_second_parent_run_id = mock_logger.log.call_args_list[1].args[1][
            "parent_run_id"
        ]
        self.assertEqual(mock_logger.log.call_args_list[1].args[1]["outputs"], 2)
        self.assertTrue("run_id" in mock_logger.log.call_args_list[1].args[1])

        # MockFunction2
        self.assertEqual(
            mock_logger.log.call_args_list[2].args[1]["name"], "MockFunction2"
        )
        self.assertTrue(
            mock_logger.log.call_args_list[2].args[1]["parent_run_id"] is not None
        )
        mock_function2_parent_run_id = mock_logger.log.call_args_list[2].args[1][
            "parent_run_id"
        ]
        mock_function2_run_id = mock_logger.log.call_args_list[2].args[1]["run_id"]
        self.assertEqual(mock_logger.log.call_args_list[2].args[1]["outputs"], 4)
        self.assertTrue("run_id" in mock_logger.log.call_args_list[2].args[1])

        # MockFunction3
        self.assertEqual(
            mock_logger.log.call_args_list[3].args[1]["name"], "MockFunction3"
        )
        self.assertTrue(
            mock_logger.log.call_args_list[3].args[1]["parent_run_id"] is None
        )
        mock_function3_run_id = mock_logger.log.call_args_list[3].args[1]["run_id"]
        self.assertEqual(mock_logger.log.call_args_list[3].args[1]["outputs"], 5)
        self.assertTrue("run_id" in mock_logger.log.call_args_list[3].args[1])

        # log inside mock_method2
        self.assertEqual(len(mock_logger.info.call_args_list), 1)
        self.assertEqual(
            mock_logger.info.call_args_list[0].args[0]["parent_run_id"],
            mock_function2_run_id,
        )

        # check lineage
        self.assertTrue(mock_function1_first_parent_run_id == mock_function2_run_id)
        self.assertTrue(mock_function1_second_parent_run_id == mock_function2_run_id)
        self.assertTrue(mock_function2_parent_run_id == mock_function3_run_id)

    def test_traceable_with_nested_functions_with_exception(self):
        mock_logger = Mock()

        @traceable("MockFunction1", mock_logger, level=logging.INFO)
        def mock_method1(number: int):
            """mock method"""

            raise Exception("test exception")

        @traceable("MockFunction2", mock_logger, level=logging.INFO)
        def mock_method2(number: int):
            """mock method"""

            return mock_method1(number) + 1

        try:
            mock_method2(1)
        except Exception:
            pass

        self.assertEqual(len(mock_logger.log.call_args_list), 2)

        # MockFunction1
        self.assertEqual(
            mock_logger.log.call_args_list[0].args[1]["name"], "MockFunction1"
        )
        self.assertTrue(
            mock_logger.log.call_args_list[0].args[1]["parent_run_id"] is not None
        )
        self.assertTrue(
            'raise Exception("test exception")'
            in mock_logger.log.call_args_list[0].args[1]["error"]
        )
        self.assertTrue("run_id" in mock_logger.log.call_args_list[0].args[1])

        # MockFunction2
        self.assertEqual(
            mock_logger.log.call_args_list[1].args[1]["name"], "MockFunction2"
        )
        self.assertTrue(
            mock_logger.log.call_args_list[1].args[1]["parent_run_id"] is None
        )
        self.assertTrue(
            'raise Exception("test exception")'
            in mock_logger.log.call_args_list[0].args[1]["error"]
        )
        self.assertTrue("run_id" in mock_logger.log.call_args_list[1].args[1])
