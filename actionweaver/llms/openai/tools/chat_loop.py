from __future__ import annotations

import itertools
import json
import logging
import time
from collections import defaultdict
from typing import List, Optional, Union

from openai import AsyncOpenAI, OpenAI, Stream
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from actionweaver.actions.action import Action, ActionHandlers
from actionweaver.llms.openai.tools.tools import Tools
from actionweaver.telemetry import traceable
from actionweaver.utils import DEFAULT_ACTION_SCOPE
from actionweaver.utils.stream import get_first_element_and_iterator, merge_dicts
from actionweaver.utils.tokens import TokenUsageTracker


class OpenAIChatCompletionException(Exception):
    def __init__(self, message="", extra_info=None):
        super().__init__(message)
        self.extra_info = extra_info or {}

    def __str__(self):
        # Customize the string representation to include extra_info
        extra_info_str = ", ".join(
            f"{key}: {value}" for key, value in self.extra_info.items()
        )
        return f"{super().__str__()} | Additional Info: [{extra_info_str}]"


class OpenAIChatCompletion:
    def __init__(self, model, token_usage_tracker=None, logger=None):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.token_usage_tracker = token_usage_tracker or TokenUsageTracker()
        self.client = OpenAI()

    @staticmethod
    def _invoke_tool(
        messages,
        model,
        response_msg,
        tool_calls,
        tools,
        orch,
        action_handler: ActionHandlers,
    ):
        messages += [response_msg]

        # if multiple type of functions are invoked, ignore orch and `stop` option
        called_tools = defaultdict(list)

        # TODO: right now invoke all tools iteratively, implement async tool invocation
        for tool_call in tool_calls:
            if isinstance(tool_call, ChatCompletionMessageToolCall):
                tool_call = tool_call.model_dump()

            name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]

            if action_handler.contains(name):
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                except json.decoder.JSONDecodeError as e:
                    raise OpenAIChatCompletionException(
                        e,
                        extra_info={
                            "message": "Parsing function call arguments from OpenAI response ",
                            "arguments": tool_call["function"]["arguments"],
                            "timestamp": time.time(),
                            "model": model,
                        },
                    ) from e

                # Invoke action
                tool_response = action_handler[name](**arguments)

                called_tools[name].append(tool_response)

                stop = action_handler[name].stop
                messages += [
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": name,
                        "content": str(tool_response),
                    },
                ]

            else:
                # TODO: allow user to add callback for unavailable tool
                unavailable_tool_msg = f"{name} is not a valid tool name, use one of the following: {', '.join([tool['function']['name'] for tool in tools.tools])}"

                raise OpenAIChatCompletionException(unavailable_tool_msg)

        if len(called_tools) == 1:
            # Update new functions for next OpenAI api call
            name = list(called_tools.keys())[0]

            # use tools in orch[DEFAULT_ACTION_SCOPE] if expr is DEFAULT_ACTION_SCOPE
            expr = (
                orch[name]
                if orch[name] != DEFAULT_ACTION_SCOPE
                else orch[DEFAULT_ACTION_SCOPE]
            )
            return (
                Tools.from_expr(
                    expr,
                ),
                (stop, called_tools[name]),
            )
        else:
            # if multiple type of functions are invoked, use the same set of tools next api call
            return (
                tools,
                (False, list(called_tools.values())),
            )

    @staticmethod
    def _handle_stream_response(api_response):
        first_element, iterator = get_first_element_and_iterator(api_response)

        if first_element.choices[0].delta.content is not None:
            # if the first element is a message, return generator right away.
            return iterator
        else:
            # if the first element is a tool call, merge all tool calls into first response and return it
            l = list(iterator)

            deltas = {}
            for element in l:
                delta = element.choices[0].delta.model_dump()
                deltas = merge_dicts(deltas, delta)

            chat_completion_message_tool_call = defaultdict(dict)
            for tool_delta in deltas["tool_calls"]:
                chat_completion_message_tool_call[tool_delta["index"]] = merge_dicts(
                    chat_completion_message_tool_call[tool_delta["index"]],
                    tool_delta,
                )
                tool_delta.pop("index")

            deltas["tool_calls"] = list(chat_completion_message_tool_call.values())

            # (HACK) Remove the 'function_call' field, otherwise calling the API will fail
            if "function_call" in deltas:
                del deltas["function_call"]

            first_element.choices[0].message = ChatCompletionMessage(**deltas)

            return first_element

    @staticmethod
    def build_orch(actions: List[Action] = None, orch=None):
        action_handler = ActionHandlers()

        if orch is None:
            orch = {}
        if DEFAULT_ACTION_SCOPE not in orch:
            orch[DEFAULT_ACTION_SCOPE] = actions

        buf = actions + list(orch.values())
        for element in buf:
            if isinstance(element, list):
                for e in element:
                    action_handler.name_to_action[e.name] = e
            elif isinstance(element, Action):
                action_handler.name_to_action[element.name] = element
        # default action scope if not following actions not specified
        for _, action in action_handler.name_to_action.items():
            if action.name not in orch:
                orch[action.name] = DEFAULT_ACTION_SCOPE

        return action_handler, orch

    def create(
        self,
        orch=None,
        actions: List[Action] = [],
        *args,
        **kwargs,
    ):
        if "model" not in kwargs:
            kwargs["model"] = self.model

        return OpenAIChatCompletion.wrap_chat_completion_create(
            self.client.chat.completions.create
        )(
            actions=actions,
            orch=orch,
            logger=self.logger,
            token_usage_tracker=self.token_usage_tracker,
            *args,
            **kwargs,
        )

    @staticmethod
    def validate_orch(orch):
        if orch is not None:
            for key in orch.keys():
                if not isinstance(key, str):
                    raise OpenAIChatCompletionException(
                        f"Orch keys must be action name (str), found {type(key)}"
                    )

    @staticmethod
    def wrap_chat_completion_create(original_create_method):
        def wrapper_for_logging(
            *args,
            logger: Optional[logging.Logger] = None,
            logging_name: Optional[str] = None,
            logging_metadata: Optional[dict] = None,
            logging_level=logging.INFO,
            **kwargs,
        ):
            DEFAULT_LOGGING_NAME = "actionweaver_initial_chat_completion"

            def new_create(
                actions: List[Action] = [],
                orch=None,
                token_usage_tracker=None,
                *args,
                **kwargs,
            ):
                OpenAIChatCompletion.validate_orch(orch)

                chat_completion_create_method = original_create_method
                if logger:
                    chat_completion_create_method = traceable(
                        name=(logging_name or DEFAULT_LOGGING_NAME)
                        + ".chat.completions.create",
                        logger=logger,
                        metadata=logging_metadata,
                        level=logging_level,
                    )(original_create_method)

                if token_usage_tracker is None:
                    token_usage_tracker = TokenUsageTracker()

                messages = kwargs.get("messages")
                if messages is None:
                    raise OpenAIChatCompletionException(
                        "messages keyword argument is required for chat completion"
                    )
                model = kwargs.get("model")
                if model is None:
                    raise OpenAIChatCompletionException(
                        "model keyword argument is required for chat completion"
                    )

                action_handler, orch = OpenAIChatCompletion.build_orch(actions, orch)
                response = None

                tools = Tools.from_expr(orch[DEFAULT_ACTION_SCOPE])

                while True:
                    tools_argument = tools.to_arguments()
                    if tools_argument["tools"]:
                        api_response = chat_completion_create_method(
                            *args,
                            **kwargs,
                            **tools_argument,
                        )
                    else:
                        api_response = chat_completion_create_method(
                            *args,
                            **kwargs,
                        )

                    # logic to handle streaming API response
                    if isinstance(api_response, Stream):
                        api_response = OpenAIChatCompletion._handle_stream_response(
                            api_response
                        )

                        if isinstance(api_response, itertools._tee):
                            # if it's a tee object, return right away
                            return api_response
                    else:
                        token_usage_tracker.track_usage(api_response.usage)

                    choice = api_response.choices[0]
                    message = choice.message

                    if message.tool_calls:
                        tools, (stop, resp) = OpenAIChatCompletion._invoke_tool(
                            messages,
                            model,
                            message,
                            message.tool_calls,
                            tools,
                            orch,
                            action_handler,
                        )
                        if stop:
                            return resp
                    elif message.content is not None:
                        response = api_response

                        # ignore last message in the function loop
                        # messages += [{"role": "assistant", "content": message["content"]}]
                        if choice.finish_reason == "stop":
                            """
                            Stop Reasons:

                            - Occurs when the API returns a message that is complete or is concluded by one of the stop sequences defined via the 'stop' parameter.

                            See https://platform.openai.com/docs/guides/gpt/chat-completions-api for details.
                            """

                            break
                    else:
                        raise OpenAIChatCompletionException(
                            f"Unsupported response from OpenAI api: {api_response}"
                        )
                return response

            if logger:
                return traceable(
                    name=logging_name or DEFAULT_LOGGING_NAME,
                    logger=logger,
                    metadata=logging_metadata,
                    level=logging_level,
                )(new_create)(*args, **kwargs)
            else:
                return new_create(*args, **kwargs)

        return wrapper_for_logging

    @staticmethod
    def patch(client: Union[OpenAI, AsyncOpenAI]):
        if isinstance(client, AsyncOpenAI):
            raise NotImplementedError(
                "AsyncOpenAI client is not supported for patching yet."
            )

        client.chat.completions.create = (
            OpenAIChatCompletion.wrap_chat_completion_create(
                client.chat.completions.create
            )
        )
        return client
