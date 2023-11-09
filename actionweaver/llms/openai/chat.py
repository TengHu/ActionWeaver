from __future__ import annotations

import json
import logging
import time
import uuid
from argparse import Action
from typing import List

from openai import OpenAI, Stream
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)

# Todo: Deprecated function_call in favor of tool_choice.


client = OpenAI()

from actionweaver.actions.action import ActionHandlers
from actionweaver.actions.orchestration import (
    Orchestration,
    build_orchestration_dict,
    parse_orchestration_expr,
)
from actionweaver.actions.orchestration_expr import (
    _ActionDefault,
    _ActionHandlerLLMInvoke,
)
from actionweaver.llms.openai.functions import Functions
from actionweaver.llms.openai.tokens import TokenUsageTracker
from actionweaver.utils import DEFAULT_ACTION_SCOPE
from actionweaver.utils.stream import get_first_element_and_iterator, merge_dicts


class OpenAIChatCompletionException(Exception):
    pass


class OpenAIChatCompletion:
    def __init__(self, model, token_usage_tracker=None, logger=None):
        self.model = model
        self.action_handlers = ActionHandlers()
        self.logger = logger or logging.getLogger(__name__)
        self.token_usage_tracker = token_usage_tracker or TokenUsageTracker(
            logger=logger
        )

    def _bind_action_handlers(
        self, action_handlers: ActionHandlers
    ) -> OpenAIChatCompletion:
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
        functions = Functions.from_expr(
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
                    **functions.to_arguments(),
                }
            )

            function_argument = functions.to_arguments()
            if function_argument["functions"]:
                api_response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    stream=stream,
                    **function_argument,
                )
            else:
                api_response = client.chat.completions.create(
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
                    import pdb

                    pdb.set_trace()
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
                    message.function_call,
                    orchestration_dict,
                    default_expr,
                    action_handler,
                    functions,
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
