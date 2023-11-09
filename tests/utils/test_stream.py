from __future__ import annotations

import unittest

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from actionweaver.utils.stream import get_first_element_and_iterator, merge_dicts


class StreamUtilsTestCase(unittest.TestCase):
    def test_get_first_element_and_iterator(self):
        gen = iter(range(3, 10))

        first_element, iter2 = get_first_element_and_iterator(gen)

        self.assertEqual(first_element, 3)
        self.assertEqual(list(iter2), list(range(3, 10)))

    # Todo: test merging function calling and tool calls streams.
    def test_merge_dicts(self):
        streams = [
            ChatCompletionChunk(
                **{
                    "id": "chatcmpl-8Izk9ayIEYUKWLhGmdpBqJOomrdpR",
                    "choices": [
                        {
                            "delta": {
                                "content": "Hello",
                                "function_call": None,
                                "role": None,
                                "tool_calls": None,
                            },
                            "finish_reason": None,
                            "index": 0,
                        }
                    ],
                    "created": 1699537937,
                    "model": "gpt-3.5-turbo-0613",
                    "object": "chat.completion.chunk",
                    "system_fingerprint": None,
                }
            ),
            ChatCompletionChunk(
                **{
                    "id": "chatcmpl-8Izk9ayIEYUKWLhGmdpBqJOomrdpR",
                    "choices": [
                        {
                            "delta": {
                                "content": "!",
                                "function_call": None,
                                "role": None,
                                "tool_calls": None,
                            },
                            "finish_reason": None,
                            "index": 0,
                        }
                    ],
                    "created": 1699537937,
                    "model": "gpt-3.5-turbo-0613",
                    "object": "chat.completion.chunk",
                    "system_fingerprint": None,
                }
            ),
            ChatCompletionChunk(
                **{
                    "id": "chatcmpl-8Izk9ayIEYUKWLhGmdpBqJOomrdpR",
                    "choices": [
                        {
                            "delta": {
                                "content": " How",
                                "function_call": None,
                                "role": None,
                                "tool_calls": None,
                            },
                            "finish_reason": None,
                            "index": 0,
                        }
                    ],
                    "created": 1699537937,
                    "model": "gpt-3.5-turbo-0613",
                    "object": "chat.completion.chunk",
                    "system_fingerprint": None,
                }
            ),
        ]

        gen = iter(streams)

        l = list(gen)

        ret = {}

        for element in l:
            delta = element.choices[0].delta.model_dump()
            ret = merge_dicts(ret, delta)

        self.assertEqual(
            ret,
            {
                "content": "Hello! How",
                "function_call": None,
                "role": None,
                "tool_calls": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
