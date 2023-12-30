from __future__ import annotations

import json
import logging
import time
import uuid
from typing import List

from openai import OpenAI, Stream
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)

from actionweaver.actions.action import Action, ActionHandlers
from actionweaver.llms.openai.functions.functions import Functions
from actionweaver.utils import DEFAULT_ACTION_SCOPE
from actionweaver.utils.stream import get_first_element_and_iterator, merge_dicts
from actionweaver.utils.tokens import TokenUsageTracker


class OpenAIChatCompletionException(Exception):
    pass


class OpenAIChatCompletion:
    def __init__(self, model, token_usage_tracker=None, logger=None):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.token_usage_tracker = token_usage_tracker or TokenUsageTracker()
        self.client = OpenAI()

        print(
            "\033[91mDeprecating soon. Please import the OpenAIChatCompletion class from actionweaver.llms.openai.tools.chat\033[0m"
        )

    def _invoke_function(
        self,
        call_id,
        messages,
        model,
        response_msg,
        function_call,
        functions,
        orch,
        action_handler,
    ):
        """Invoke the function, update the messages, returns functions argument for the next OpenAI API call or halt the function loop and return the response."""

        if isinstance(function_call, FunctionCall):
            function_call = function_call.model_dump()

        messages += [response_msg]

        name = function_call["name"]
        arguments = function_call["arguments"]

        if action_handler.contains(name):
            try:
                arguments = json.loads(function_call["arguments"])
            except json.decoder.JSONDecodeError as e:
                self.logger.error(
                    {
                        "message": "Parsing function call arguments from OpenAI response ",
                        "arguments": function_call["arguments"],
                        "timestamp": time.time(),
                        "model": model,
                        "call_id": call_id,
                    },
                    exc_info=True,
                )
                raise OpenAIChatCompletionException(e) from e

            # Invoke action
            function_response = action_handler[name](**arguments)
            stop = action_handler[name].stop
            messages += [
                {
                    "role": "function",
                    "name": name,
                    "content": str(function_response),
                }
            ]

            self.logger.debug(
                {
                    "message": "Action invoked and response received",
                    "action_name": name,
                    "action_arguments": arguments,
                    "action_response": function_response,
                    "timestamp": time.time(),
                    "call_id": call_id,
                    "stop": stop,
                }
            )

            expr = (
                orch[action_handler[name]]
                if orch[action_handler[name]] != DEFAULT_ACTION_SCOPE
                else orch[DEFAULT_ACTION_SCOPE]
            )

            return (
                Functions.from_expr(
                    expr,
                ),
                (stop, function_response),
            )
        else:
            unavailable_function_msg = f"{name} is not a valid function name, use one of the following: {', '.join([func['name'] for func in functions.functions])}"
            messages += [
                {
                    "role": "user",
                    "content": unavailable_function_msg,
                }
            ]
            self.logger.debug(
                {
                    "message": "Unavailable action",
                    "action_name": name,
                    "action_arguments": arguments,
                    "action_response": unavailable_function_msg,
                    "timestamp": time.time(),
                    "call_id": call_id,
                }
            )
            return functions, (False, None)

    def create(
        self,
        messages,
        *args,
        orch=None,
        stream=False,
        actions: List[Action] = [],
        **kwargs,
    ):
        """
        Invoke the OpenAI API with the provided messages and functions.

        Functions supplied to the API are determined by the specified scope and orchestration expression.

        Args:
            messages (list): List of message objects for the conversation.
            scope (str): Scope of the functions to be used.
            stream (bool): If True, returns a generator that yields the API responses.

        Returns:
            API response with generated output.
        """

        # Todo: pass call_id to the decorated method
        call_id = str(uuid.uuid4())

        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", 0.0)

        # Restart token usage tracker
        self.token_usage_tracker.clear()

        # TODO: add validation to orch,
        action_handler = ActionHandlers()

        if orch is None:
            orch = {}
        if DEFAULT_ACTION_SCOPE not in orch:
            orch[DEFAULT_ACTION_SCOPE] = actions

        buf = actions + list(orch.keys()) + list(orch.values())

        for element in buf:
            if isinstance(element, list):
                for e in element:
                    action_handler.name_to_action[e.name] = e
            elif isinstance(element, Action):
                action_handler.name_to_action[element.name] = element
        # default action scope if not following actions not specified
        for _, action in action_handler.name_to_action.items():
            if action not in orch:
                orch[action] = DEFAULT_ACTION_SCOPE

        self.logger.debug(
            {
                "message": "Creating new chat completion",
                "model": model,
                "input_messages": messages,
                "timestamp": time.time(),
                "call_id": call_id,
            }
        )

        response = None
        functions = Functions.from_expr(orch[DEFAULT_ACTION_SCOPE])

        while True:
            self.logger.debug(
                {
                    "message": "Calling OpenAI API",
                    "call_id": call_id,
                    "input_messages": messages,
                    "model": model,
                    "timestamp": time.time(),
                    **functions.to_arguments(),
                }
            )

            function_argument = functions.to_arguments()
            if function_argument["functions"]:
                api_response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    stream=stream,
                    **function_argument,
                )
            else:
                api_response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    stream=stream,
                )

            # logic to handle streaming API response
            if isinstance(api_response, Stream):
                first_element, iterator = get_first_element_and_iterator(api_response)

                if first_element.choices[0].delta.content is not None:
                    # if the first element is a message, return generator right away.
                    return iterator
                elif first_element.choices[0].delta.function_call:
                    # if the first element in generator is a function call, merge all the deltas.
                    l = list(iterator)

                    deltas = {}
                    for element in l:
                        delta = element.choices[0].delta.model_dump()
                        deltas = merge_dicts(deltas, delta)

                    first_element.choices[0].message = ChatCompletionMessage(**deltas)
                    api_response = first_element
                else:
                    raise OpenAIChatCompletionException(
                        f"Unsupported response from streaming API: {list(iterator)}"
                    )

            else:
                self.token_usage_tracker.track_usage(api_response.usage)

            self.logger.debug(
                {
                    "message": "Received response from OpenAI API",
                    "response": api_response,
                    "timestamp": time.time(),
                    "call_id": call_id,
                }
            )

            choice = api_response.choices[0]
            message = choice.message

            if message.function_call:
                functions, (stop, resp) = self._invoke_function(
                    call_id,
                    messages,
                    model,
                    message,
                    message.function_call,
                    functions,
                    orch,
                    action_handler,
                )
                if stop:
                    return resp
            elif message.content is not None:
                response = message.content

                # ignore last message in the function loop
                # messages += [{"role": "assistant", "content": message["content"]}]
                if choice.finish_reason == "stop":
                    """
                    Stop Reasons:

                    - Occurs when the API returns a message that is complete or is concluded by one of the stop sequences defined via the 'stop' parameter.

                    See https://platform.openai.com/docs/guides/gpt/chat-completions-api for details.
                    """
                    self.logger.debug(
                        {
                            "message": "Model decides to stop",
                            "model": model,
                            "timestamp": time.time(),
                            "call_id": call_id,
                        }
                    )

                    break
            else:
                raise OpenAIChatCompletionException(
                    f"Unsupported response from OpenAI api: {api_response}"
                )
        return response
