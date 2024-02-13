from __future__ import annotations

import itertools
import json
import logging
import time
import uuid
from itertools import chain
from typing import List, Optional

from openai import Stream
from openai.types.chat.chat_completion_message import FunctionCall

import actionweaver.llms.loop_action as la
from actionweaver.actions.action import Action, ActionHandlers
from actionweaver.llms.azure.functions import Functions
from actionweaver.llms.exception_handler import ChatLoopInfo, ExceptionHandler
from actionweaver.telemetry import traceable
from actionweaver.utils import DEFAULT_ACTION_SCOPE
from actionweaver.utils.stream import get_first_element_and_iterator
from actionweaver.utils.tokens import TokenUsageTracker

# TODO: support AsyncAzureOpenAI


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


def handle_stream_response(api_response):
    _, iterator = get_first_element_and_iterator(api_response)
    stream_function_calls = None
    for chunk in iterator:
        if chunk.choices:
            delta = chunk.choices[0].delta

            if delta.function_call:
                if stream_function_calls is None:
                    # if function call detected, we merge all deltas and treat it as the non-stream response
                    stream_function_calls = FunctionCall(name="", arguments="")

                stream_function_calls.name += (
                    delta.function_call.name if delta.function_call.name else ""
                )

                stream_function_calls.arguments += (
                    delta.function_call.arguments
                    if delta.function_call.arguments
                    else ""
                )

            elif delta.content:
                # if it has content return as generator right away
                return chain([chunk], iterator)
            else:
                raise FunctionCallingLoopException(
                    f"Unsupported streaming response",
                    extra_info={
                        "message": "Unsupported streaming response",
                        "timestamp": time.time(),
                    },
                )
    return stream_function_calls


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


def invoke_function(
    messages,
    model,
    function_call,
    functions,
    orch,
    action_handler,
):
    """Invoke the function, update the messages, returns functions argument for the next OpenAI API call or halt the function loop and return the response."""

    if isinstance(function_call, FunctionCall):
        function_call = function_call.model_dump()

    messages += [
        {
            "role": "assistant",
            "content": None,
            "function_call": function_call,
        }
    ]

    name = function_call["name"]
    arguments = function_call["arguments"]

    if action_handler.contains(name):
        try:
            arguments = json.loads(function_call["arguments"])
        except json.decoder.JSONDecodeError as e:
            raise FunctionCallingLoopException(
                "Parsing function call arguments from OpenAI response failed",
                extra_info={
                    "arguments": function_call["arguments"],
                    "timestamp": time.time(),
                    "model": model,
                },
            ) from e

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

        # use tools in orch[DEFAULT_ACTION_SCOPE] if expr is DEFAULT_ACTION_SCOPE
        expr = (
            orch[name]
            if orch[name] != DEFAULT_ACTION_SCOPE
            else orch[DEFAULT_ACTION_SCOPE]
        )
        return (
            Functions.from_expr(
                expr,
            ),
            (stop, function_response),
        )
    else:
        raise FunctionCallingLoopException(
            f"{name} is not a valid function name",
            extra_info={
                "timestamp": time.time(),
                "model": model,
            },
        )


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

            functions = Functions.from_expr(orch[DEFAULT_ACTION_SCOPE])

            while True:

                function_argument = functions.to_arguments()
                chat_loop_action = la.Unknown

                try:
                    if functions:
                        api_response = chat_completion_create_method(
                            *args,
                            **kwargs,
                            **function_argument,
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
                        functions,
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
                                    "messages": messages,
                                    "functions": functions,
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
                    functions = chat_loop_action.functions
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

    if "functions" in kwargs:
        raise FunctionCallingLoopException(
            "functions keyword argument is not allowed for this method, use actions instead"
        )

    if "function_call" in kwargs:
        raise FunctionCallingLoopException(
            "function_call keyword argument is not allowed for this method, use actions instead"
        )


def handle_response(
    api_response,
    token_usage_tracker,
    messages,
    model,
    functions,
    orch,
    action_handler,
    logger=None,
) -> la.LoopAction:
    # logic to handle streaming API response
    processed_stream_response = None
    if isinstance(api_response, Stream):
        processed_stream_response = handle_stream_response(api_response, logger)

        if type(processed_stream_response) == itertools.chain:
            # if it's a tee object, return the message right away
            return processed_stream_response
    else:
        token_usage_tracker.track_usage(api_response.usage)

    if processed_stream_response is not None:
        functions, (
            stop,
            resp,
        ) = invoke_function(
            messages,
            model,
            processed_stream_response,
            functions,
            orch,
            action_handler,
        )
        if stop:
            return la.ReturnRightAway(content=resp)
        else:
            return la.Continue(functions=functions)
    else:
        choice = api_response.choices[0]
        message = choice.message

        if message.function_call:
            functions, (
                stop,
                resp,
            ) = invoke_function(
                messages,
                model,
                message.function_call,
                functions,
                orch,
                action_handler,
            )
            if stop:
                return la.ReturnRightAway(content=resp)
            else:
                return la.Continue(functions=functions)
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
