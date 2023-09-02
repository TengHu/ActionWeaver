from __future__ import annotations

import unittest

from openai.openai_object import OpenAIObject

from actionweaver.utils.stream import get_first_element_and_iterator, merge_dicts


class StreamUtilsTestCase(unittest.TestCase):
    def test_get_first_element_and_iterator(self):
        gen = iter(range(3, 10))

        first_element, iter2 = get_first_element_and_iterator(gen)

        self.assertEqual(first_element, 3)
        self.assertEqual(list(iter2), list(range(3, 10)))

    def test_merge_dicts(self):
        streams = [
            OpenAIObject.construct_from(
                {
                    "id": "chatcmpl-7uPkC4A9XVJEyQW3drBGhmOWmFsZ0",
                    "object": "chat.completion.chunk",
                    "created": 1693679684,
                    "model": "gpt-3.5-turbo-0613",
                    "choices": [
                        OpenAIObject.construct_from(
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": None,
                                    "function_call": {
                                        "name": "GetCurrentTime",
                                        "arguments": "",
                                    },
                                },
                                "finish_reason": None,
                            }
                        )
                    ],
                }
            ),
            OpenAIObject.construct_from(
                {
                    "id": "chatcmpl-7uPkC4A9XVJEyQW3drBGhmOWmFsZ0",
                    "object": "chat.completion.chunk",
                    "created": 1693679684,
                    "model": "gpt-3.5-turbo-0613",
                    "choices": [
                        OpenAIObject.construct_from(
                            {
                                "index": 0,
                                "delta": {"function_call": {"arguments": "{\n\n"}},
                                "finish_reason": None,
                            }
                        )
                    ],
                }
            ),
            OpenAIObject.construct_from(
                {
                    "id": "chatcmpl-7uPkC4A9XVJEyQW3drBGhmOWmFsZ0",
                    "object": "chat.completion.chunk",
                    "created": 1693679684,
                    "model": "gpt-3.5-turbo-0613",
                    "choices": [
                        OpenAIObject.construct_from(
                            {
                                "index": 0,
                                "delta": {"function_call": {"arguments": "}"}},
                                "finish_reason": None,
                            }
                        ),
                    ],
                }
            ),
        ]

        gen = iter(streams)

        l = list(gen)

        ret = {}
        for element in l:
            delta = element["choices"][0]["delta"].to_dict()
            ret = merge_dicts(ret, delta)

        self.assertEqual(
            ret,
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "GetCurrentTime", "arguments": "{\n\n}"},
            },
        )


if __name__ == "__main__":
    unittest.main()
