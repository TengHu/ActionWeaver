from __future__ import annotations

import itertools
import json
import logging
import time
import uuid
from itertools import chain
from typing import List, Optional, Union

from openai import AsyncAzureOpenAI, AzureOpenAI, Stream
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)

from actionweaver.actions.action import Action, ActionHandlers
from actionweaver.llms.azure.functions import Functions
from actionweaver.telemetry import traceable
from actionweaver.utils import DEFAULT_ACTION_SCOPE
from actionweaver.utils.stream import get_first_element_and_iterator, merge_dicts
from actionweaver.utils.tokens import TokenUsageTracker

# TODO: support AsyncAzureOpenAI


class ChatCompletionException(Exception):
    def __init__(self, message="", extra_info=None):
        super().__init__(message)
        self.extra_info = extra_info or {}

    def __str__(self):
        # Customize the string representation to include extra_info
        extra_info_str = ", ".join(
            f"{key}: {value}" for key, value in self.extra_info.items()
        )
        return f"{super().__str__()} | Additional Info: [{extra_info_str}]"


class ChatCompletion:
    def __init__(
        self,
        model,
        azure_endpoint,
        api_key,
        api_version,
        token_usage_tracker=None,
        logger=None,
        azure_deployment="",
    ):
        self.model = model
        self.action_handlers = ActionHandlers()
        self.logger = logger or logging.getLogger(__name__)
        if azure_deployment == "":
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version
            )
        else:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                azure_deployment=azure_deployment,
            )
        self.token_usage_tracker = token_usage_tracker or TokenUsageTracker()

    @staticmethod
    def _handle_stream_response(api_response):
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
                    raise ChatCompletionException(
                        f"Unsupported streaming response",
                        extra_info={
                            "message": "Unsupported streaming response",
                            "timestamp": time.time(),
                        },
                    )
        return stream_function_calls

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

    @staticmethod
    def _invoke_function(
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
                raise ChatCompletionException(
                    e,
                    extra_info={
                        "message": "Parsing function call arguments from OpenAI response ",
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
            unavailable_function_msg = f"{name} is not a valid function name, use one of the following: {', '.join([func['name'] for func in functions.functions])}"
            messages += [
                {
                    "role": "user",
                    "content": unavailable_function_msg,
                }
            ]
            return functions, (False, None)

    def create(
        self,
        orch=None,
        actions: List[Action] = [],
        *args,
        **kwargs,
    ):
        if "model" not in kwargs:
            kwargs["model"] = self.model

        return ChatCompletion.wrap_chat_completion_create(
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
                    raise ChatCompletionException(
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
                ChatCompletion.validate_orch(orch)

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
                    raise ChatCompletionException(
                        "messages keyword argument is required for chat completion"
                    )
                model = kwargs.get("model")
                if model is None:
                    raise ChatCompletionException(
                        "model keyword argument is required for chat completion"
                    )

                action_handler, orch = ChatCompletion.build_orch(actions, orch)

                response = None
                functions = Functions.from_expr(orch[DEFAULT_ACTION_SCOPE])

                while True:
                    function_argument = functions.to_arguments()
                    if function_argument["functions"]:
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

                    # logic to handle streaming API response
                    processed_stream_response = None
                    if isinstance(api_response, Stream):
                        processed_stream_response = (
                            ChatCompletion._handle_stream_response(api_response, logger)
                        )

                        if type(processed_stream_response) == itertools.chain:
                            # if it's a tee object, return the message right away
                            return processed_stream_response
                    else:
                        token_usage_tracker.track_usage(api_response.usage)

                    if processed_stream_response is not None:
                        functions, (
                            stop,
                            resp,
                        ) = ChatCompletion._invoke_function(
                            messages,
                            model,
                            processed_stream_response,
                            functions,
                            orch,
                            action_handler,
                        )
                        if stop:
                            return resp
                    else:
                        choice = api_response.choices[0]
                        message = choice.message

                        if message.function_call:
                            functions, (
                                stop,
                                resp,
                            ) = ChatCompletion._invoke_function(
                                messages,
                                model,
                                message.function_call,
                                functions,
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
                            raise ChatCompletionException(
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
    def patch(client: Union[AzureOpenAI, AsyncAzureOpenAI]):
        if isinstance(client, AsyncAzureOpenAI):
            raise NotImplementedError(
                "AsyncAzureOpenAI client is not supported for patching yet."
            )

        client.chat.completions.create = ChatCompletion.wrap_chat_completion_create(
            client.chat.completions.create
        )
        return client
