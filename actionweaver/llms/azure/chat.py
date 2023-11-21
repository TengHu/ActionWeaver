from __future__ import annotations

import inspect
import json
import logging
import time
import uuid
from argparse import Action
from itertools import chain
from typing import List

from openai import AzureOpenAI, Stream
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)

from actionweaver.actions.action import ActionHandlers
from actionweaver.actions.orchestration import (
    build_orchestration_dict,
    parse_orchestration_expr,
)
from actionweaver.actions.orchestration_expr import (
    _ActionDefault,
    _ActionHandlerLLMInvoke,
)
from actionweaver.llms.azure.functions import Functions
from actionweaver.llms.azure.tokens import TokenUsageTracker
from actionweaver.utils import DEFAULT_ACTION_SCOPE
from actionweaver.utils.stream import get_first_element_and_iterator, merge_dicts

# TODO: support AsyncAzureOpenAI


class ChatCompletionException(Exception):
    pass


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
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
        else:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                azure_deployment=azure_deployment
            )
        self.token_usage_tracker = token_usage_tracker or TokenUsageTracker(
            logger=logger
        )

    def _bind_action_handlers(self, action_handlers: ActionHandlers) -> ChatCompletion:
        self.action_handlers = action_handlers
        return self

    def _invoke_function(
        self,
        call_id,
        messages,
        model,
        function_call,
        orchestration_dict,
        default_expr,
        action_handler: ActionHandlers,
        functions,
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
                raise ChatCompletionException(e) from e

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

            # Update new functions for next OpenAI api call
            # if name doesn't exist in orchestration dict, use _ActionDefaultLLM which doesn't invoke functions
            expr = (
                orchestration_dict[name]
                if name in orchestration_dict
                else _ActionDefault()
            )
            return (
                Functions.from_expr(
                    expr,
                    action_handler,
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
        scope=None,
        orch_expr=None,
        stream=False,
        action_handler: ActionHandlers = None,
        actions: List[Action] = [],
        **kwargs,
    ):
        """
        Invoke the OpenAI API with the provided messages and functions.

        Functions supplied to the API are determined by the specified scope and orchestration expression.

        Args:
            messages (list): List of message objects for the conversation.
            scope (str): Scope of the functions to be used.
            orch_expr: If specified, overrides the orchestration dictionary, determining the API call flow.
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

        if scope is None:
            scope = DEFAULT_ACTION_SCOPE
        default_expr = _ActionHandlerLLMInvoke(scope)

        # TODO: move below into init method
        # initialize action handlers
        if action_handler is None:
            if actions:
                action_handler = ActionHandlers.from_actions(actions)
            else:
                action_handler = self.action_handlers

        for _, action in action_handler.name_to_action.items():
            action_handler.check_orchestration_expr_validity(action.orch_expr)

        # Build orchestration
        orchestration_dict = build_orchestration_dict(action_handler)

        # If an orchestration expression is provided, override the orchestration dictionary
        if orch_expr is not None:
            action_handler.check_orchestration_expr_validity(orch_expr)

            scope = "_llm_orchestration"
            # Construct a new orchestration dictionary using the parsed orchestration expression
            new_orchestration_dict = parse_orchestration_expr([scope] + orch_expr)

            # Replace the default expression with the new one
            default_expr = _ActionHandlerLLMInvoke(scope)
            new_orchestration_dict[default_expr] = new_orchestration_dict.pop(scope)

            # Update the orchestration dictionary with the new one
            orchestration_dict = new_orchestration_dict

        self.logger.debug(
            {
                "message": "Creating new chat completion",
                "model": model,
                "scope": scope,
                "input_messages": messages,
                "timestamp": time.time(),
                "call_id": call_id,
            }
        )

        response = None
        functions_to_be_called_with = Functions.from_expr(
            orchestration_dict[default_expr],
            action_handler,
        )

        while True:
            self.logger.debug(
                {
                    "message": "Calling OpenAI API",
                    "call_id": call_id,
                    "input_messages": messages,
                    "model": model,
                    "scope": scope,
                    "timestamp": time.time(),
                    **functions_to_be_called_with.to_arguments(),
                }
            )

            function_argument = functions_to_be_called_with.to_arguments()

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

            self.logger.debug(
                {
                    "message": "Received response from OpenAI API",
                    "response": api_response,
                    "timestamp": time.time(),
                    "call_id": call_id,
                }
            )

            # logic to handle streaming API response
            stream_function_calls = None
            if isinstance(api_response, Stream):
                _, iterator = get_first_element_and_iterator(api_response)

                # TODO: move this to open ai as well

                for chunk in iterator:
                    if chunk.choices:
                        delta = chunk.choices[0].delta

                        if delta.function_call:
                            if stream_function_calls is None:
                                # if function call detected, we merge all deltas and treat it as the non-stream response
                                stream_function_calls = FunctionCall(
                                    name="", arguments=""
                                )

                            stream_function_calls.name += (
                                delta.function_call.name
                                if delta.function_call.name
                                else ""
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
                            self.logger.debug(
                                {
                                    "message": "Unsupported streaming response",
                                    "chunk": str(chunk),
                                    "timestamp": time.time(),
                                    "call_id": call_id,
                                }
                            )

            else:
                self.token_usage_tracker.track_usage(api_response.usage)

            if stream_function_calls is not None:
                functions_to_be_called_with, (stop, resp) = self._invoke_function(
                    call_id,
                    messages,
                    model,
                    stream_function_calls,
                    orchestration_dict,
                    default_expr,
                    action_handler,
                    functions_to_be_called_with,
                )
                if stop:
                    return resp
            else:
                choice = api_response.choices[0]
                message = choice.message

                if message.function_call:
                    functions_to_be_called_with, (stop, resp) = self._invoke_function(
                        call_id,
                        messages,
                        model,
                        message.function_call,
                        orchestration_dict,
                        default_expr,
                        action_handler,
                        functions_to_be_called_with,
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
                    raise ChatCompletionException(
                        f"Unsupported response from OpenAI api: {api_response}"
                    )
        return response