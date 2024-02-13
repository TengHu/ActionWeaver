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

import actionweaver.llms.loop_action as la
from actionweaver.actions.action import Action, ActionHandlers
from actionweaver.llms.exception_handler import ChatLoopInfo, ExceptionHandler
from actionweaver.llms.openai.tools.tools import Tools
from actionweaver.telemetry import traceable
from actionweaver.utils import DEFAULT_ACTION_SCOPE
from actionweaver.utils.stream import get_first_element_and_iterator, merge_dicts
from actionweaver.utils.tokens import TokenUsageTracker


class FunctionCallingLoopException(Exception):
    def __init__(self, message="", extra_info=None):
        super().__init__(message)
        self.extra_info = extra_info or {}

    def __str__(self):
        # Customize the string representation to include extra_info
        extra_info_str = ", ".join(
            f"{key}: {value}" for key, value in self.extra_info.items()
        )
        return f"{super().__str__()} | Additional Info: [{extra_info_str}]"


def invoke_tool(
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
                raise FunctionCallingLoopException(
                    f"Failed to parse function call arguments from OpenAI response",
                    extra_info={
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
            raise FunctionCallingLoopException(
                f"{name} is not a valid function name",
                extra_info={
                    "timestamp": time.time(),
                    "model": model,
                },
            )

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


def handle_stream_response(api_response):
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


def validate_orch(orch):
    if orch is not None:
        for key in orch.keys():
            if not isinstance(key, str):
                raise FunctionCallingLoopException(
                    f"Orch keys must be action name (str), found {type(key)}"
                )


def create_chat_loop(original_create_method):
    def wrapper_for_logging(
        *args,
        logger: Optional[logging.Logger] = None,
        logging_name: Optional[str] = None,
        logging_metadata: Optional[dict] = None,
        logging_level=logging.INFO,
        exception_handler: ExceptionHandler = None,
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
            validate_orch(orch)

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

            argument_check(*args, **kwargs)

            messages = kwargs.get("messages")
            model = kwargs.get("model")

            action_handler, orch = build_orch(actions, orch)

            tools = Tools.from_expr(orch[DEFAULT_ACTION_SCOPE])
            chat_loop_action = la.Unknown

            while True:

                try:
                    if bool(tools):
                        tools_argument = tools.to_arguments()
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

                    chat_loop_action = handle_response(
                        api_response,
                        token_usage_tracker,
                        messages,
                        model,
                        tools,
                        orch,
                        action_handler,
                        logger,
                    )
                except Exception as e:
                    if exception_handler:
                        chat_loop_action = exception_handler.handle_exception(
                            e,
                            ChatLoopInfo(
                                context={
                                    "response": api_response,
                                    "tools": tools,
                                    "messages": messages,
                                    "model": model,
                                    "orch": orch,
                                }
                            ),
                        )

                    else:
                        raise e

                if isinstance(chat_loop_action, la.ReturnRightAway):
                    return chat_loop_action.content
                elif isinstance(chat_loop_action, la.Continue):
                    tools = chat_loop_action.functions
                else:
                    raise FunctionCallingLoopException(
                        f"Unsupported chat loop action: {chat_loop_action}"
                    )

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


def argument_check(
    *args,
    **kwargs,
):
    if "messages" not in kwargs:
        raise FunctionCallingLoopException(
            "messages keyword argument is required for chat completion"
        )
    if "model" not in kwargs:
        raise FunctionCallingLoopException(
            "model keyword argument is required for chat completion"
        )

    if "tools" in kwargs:
        raise FunctionCallingLoopException(
            "tools keyword argument is not allowed for this method, use actions instead"
        )

    if "tool_choice" in kwargs:
        raise FunctionCallingLoopException(
            "tool_choice keyword argument is not allowed for this method, use actions instead"
        )


def handle_response(
    api_response,
    token_usage_tracker,
    messages,
    model,
    tools,
    orch,
    action_handler,
    logger=None,
) -> la.LoopAction:

    # logic to handle streaming API response
    if isinstance(api_response, Stream):
        api_response = handle_stream_response(api_response)

        if isinstance(api_response, itertools._tee):
            # if it's a tee object, return right away
            return la.ReturnRightAway(content=api_response)
    else:
        token_usage_tracker.track_usage(api_response.usage)

    choice = api_response.choices[0]
    message = choice.message

    if message.tool_calls:
        tools, (stop, resp) = invoke_tool(
            messages,
            model,
            message,
            message.tool_calls,
            tools,
            orch,
            action_handler,
        )
        if stop:
            return la.ReturnRightAway(content=resp)
        else:
            return la.Continue(functions=tools)
    elif message.content is not None:

        # ignore last message in the function loop
        # messages += [{"role": "assistant", "content": message["content"]}]
        if choice.finish_reason == "stop":
            """
            Stop Reasons:

            - Occurs when the API returns a message that is complete or is concluded by one of the stop sequences defined via the 'stop' parameter.

            See https://platform.openai.com/docs/guides/gpt/chat-completions-api for details.
            """

            return la.ReturnRightAway(content=api_response)
    else:
        raise FunctionCallingLoopException(
            f"Unsupported response from OpenAI api: {api_response}"
        )
