import json
import logging
import time
import uuid

import openai

from actionweaver.actions.action import (
    ActionHandlers,
    InstanceActionHandlers,
    parse_orchestration_expr,
)
from actionweaver.actions.orchestration import _ActionHandlerLLMInvoke
from actionweaver.llms.openai.functions import Functions
from actionweaver.llms.openai.tokens import TokenUsageTracker
from actionweaver.utils import DEFAULT_ACTION_SCOPE


class OpenAIChatCompletionException(Exception):
    pass


class OpenAIChatCompletion:
    def __init__(self, model, token_usage_tracker=None, logger=None):
        self.model = model
        self.instance_action_handlers = InstanceActionHandlers(
            None, ActionHandlers()
        ).build_orchestration_dict()
        self.logger = logger or logging.getLogger(__name__)
        self.token_usage_tracker = token_usage_tracker or TokenUsageTracker(
            logger=logger
        )

    def _bind_action_handlers(self, instance_action_handlers: InstanceActionHandlers):
        self.instance_action_handlers = instance_action_handlers

    def _invoke_function(
        self,
        call_id,
        messages,
        function_call,
        orchestration_dict,
        default_expr,
        functions,
    ):
        """Invoke the API with function provided, update the messages, returns functions for the next OpenAI API call."""
        messages += [
            {
                "role": "assistant",
                "content": None,
                "function_call": function_call.to_dict(),
            }
        ]

        name = function_call["name"]
        arguments = function_call["arguments"]

        if self.instance_action_handlers.contains(name):
            try:
                arguments = json.loads(function_call["arguments"])
            except json.decoder.JSONDecodeError as e:
                self.logger.error(
                    {
                        "message": "Parsing function call arguments from OpenAI response ",
                        "arguments": function_call["arguments"],
                        "timestamp": time.time(),
                        "model": self.model,
                        "call_id": call_id,
                    },
                    exc_info=True,
                )
                raise OpenAIChatCompletionException(e) from e

            # Invoke action
            function_response = self.instance_action_handlers[name](**arguments)
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
                }
            )

            # Update new functions for next OpenAI api call
            # if name doesn't exist in orchestration dict, use default_expr
            expr = (
                orchestration_dict[name]
                if name in orchestration_dict
                else orchestration_dict[default_expr]
            )
            return Functions.from_expr(
                expr,
                self.instance_action_handlers,
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
            return functions

    def create(self, messages, *args, scope=None, orch_expr=None, **kwargs):
        """
        Invoke the OpenAI API with the provided messages and functions.

        Functions supplied to the API are determined by the specified scope and orchestration expression.

        Args:
            messages (list): List of message objects for the conversation.
            scope (str): Scope of the functions to be used.
            orch_expr: If specified, overrides the orchestration dictionary, determining the API call flow.

        Returns:
            API response with generated output.
        """
        call_id = str(uuid.uuid4())

        # Restart token usage tracker
        self.token_usage_tracker.clear()

        if scope is None:
            scope = DEFAULT_ACTION_SCOPE
        default_expr = _ActionHandlerLLMInvoke(scope)

        orchestration_dict = self.instance_action_handlers.orch_dict
        # If an orchestration expression is provided, override the orchestration dictionary
        if orch_expr is not None:
            self.instance_action_handlers.action_handlers.check_orchestration_expr_validity(
                orch_expr
            )

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
                "model": self.model,
                "scope": scope,
                "input_messages": messages,
                "timestamp": time.time(),
                "call_id": call_id,
            }
        )

        response = None
        functions = Functions.from_expr(
            orchestration_dict[default_expr],
            self.instance_action_handlers,
        )

        while True:
            self.logger.debug(
                {
                    "message": "Calling OpenAI API",
                    "call_id": call_id,
                    "input_messages": messages,
                    "model": self.model,
                    "scope": scope,
                    "timestamp": time.time(),
                    **functions.to_arguments(),
                }
            )

            function_argument = functions.to_arguments()
            if function_argument["functions"]:
                api_response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    *args,
                    **function_argument,
                    **kwargs,
                )
            else:
                api_response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    *args,
                    **kwargs,
                )

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
            message = choice["message"]

            if "function_call" in message and message["function_call"]:
                functions = self._invoke_function(
                    call_id,
                    messages,
                    message["function_call"],
                    orchestration_dict,
                    default_expr,
                    functions,
                )
            elif "content" in message and message["content"]:
                response = message["content"]
                messages += [{"role": "assistant", "content": message["content"]}]
                if choice["finish_reason"] == "stop":
                    """
                    Stop Reasons:

                    - Occurs when the API returns a message that is complete or is concluded by one of the stop sequences defined via the 'stop' parameter.

                    See https://platform.openai.com/docs/guides/gpt/chat-completions-api for details.
                    """
                    self.logger.debug(
                        {
                            "message": "Model decides to stop",
                            "model": self.model,
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
