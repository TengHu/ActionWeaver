import json
import logging
import time

import openai

from actionweaver.action import ActionHandlers
from actionweaver.llms.openai.tokens import TokenUsageTracker


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

    def _bind_action_handlers(self, action_handlers: ActionHandlers):
        self.action_handlers = action_handlers

    def create(self, messages, *args, scope=None, **kwargs):
        # Restart token usage tracker
        self.token_usage_tracker.clear()

        if scope is None:
            scope = "global"

        self.logger.debug(
            {
                "message": "Creating new chat completion",
                "model": self.model,
                "scope": scope,
                "input_messages": messages,
                "timestamp": time.time(),
            }
        )

        functions = [
            {
                "name": action.name,
                "description": action.description,
                "parameters": action.json_schema(),
            }
            for _, action in self.action_handlers.scope(scope).items()
        ]

        if len(functions) == 0:
            self.logger.debug(
                {
                    "message": "Calling OpenAI api without functions",
                    "input_messages": messages,
                    "timestamp": time.time(),
                }
            )
            api_response = openai.ChatCompletion.create(
                model=self.model, messages=messages, *args, **kwargs
            )
            self.token_usage_tracker.track_usage(api_response.usage)

            self.logger.debug(
                {
                    "message": "Received response from OpenAI api",
                    "response": api_response,
                    "model": self.model,
                    "timestamp": time.time(),
                }
            )
            message = api_response.choices[0]["message"]

            if "content" in message and message["content"]:
                response = message["content"]
                messages += [{"role": "assistant", "content": message["content"]}]
                self.logger.debug(
                    {
                        "message": "Processing message response",
                        "role": message["role"],
                        "content": message["content"],
                        "model": self.model,
                        "timestamp": time.time(),
                    }
                )
            else:
                self.logger.error(
                    {
                        "message": "Unsupported response from OpenAI api",
                        "timestamp": time.time(),
                    },
                    exc_info=True,
                )

                raise OpenAIChatCompletionException(
                    "Unsupported response from OpenAI api"
                )
            return response
        else:
            return self.function_dispatch_loop(
                messages, functions, *args, scope=scope, **kwargs
            )

    def function_dispatch_loop(self, messages, functions, *args, scope=None, **kwargs):
        response = None
        while True:
            self.logger.debug(
                {
                    "message": "Calling OpenAI api with functions",
                    "input_messages": messages,
                    "model": self.model,
                    "scope": scope,
                    "timestamp": time.time(),
                    "functions": [func["name"] for func in functions],
                }
            )

            api_response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                functions=functions,
                *args,
                **kwargs,
            )

            self.token_usage_tracker.track_usage(api_response.usage)

            self.logger.debug(
                {
                    "message": "Received response from OpenAI api",
                    "response": api_response,
                    "timestamp": time.time(),
                }
            )

            message = api_response.choices[0]["message"]

            if "content" in message and message["content"]:
                response = message["content"]
                messages += [{"role": "assistant", "content": message["content"]}]
                self.logger.debug(
                    {
                        "message": "Processing message response",
                        "role": message["role"],
                        "content": message["content"],
                        "model": self.model,
                        "timestamp": time.time(),
                    }
                )
                break
            elif "function_call" in message and message["function_call"]:
                messages += [
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": message["function_call"].to_dict(),
                    }
                ]
                name = message["function_call"]["name"]
                arguments = message["function_call"]["arguments"]

                self.logger.debug(
                    {
                        "message": "Processing function call",
                        "model": self.model,
                        "function_name": name,
                        "function_arguments": arguments,
                        "timestamp": time.time(),
                    }
                )

                if self.action_handlers.contains(name):
                    try:
                        arguments = json.loads(message["function_call"]["arguments"])
                    except json.decoder.JSONDecodeError as e:
                        self.logger.error(
                            {
                                "message": "Parsing function call arguments from OpenAi response ",
                                "arguments": message["function_call"]["arguments"],
                                "timestamp": time.time(),
                                "model": self.model,
                            },
                            exc_info=True,
                        )
                        raise OpenAIChatCompletionException(e) from e

                    # Invoke action
                    function_response = self.action_handlers[name](**arguments)
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
                        }
                    )
                else:
                    unavailable_function_msg = f"{name} is not a valid function name, use one of the following: {', '.join([func['name'] for func in functions])}"
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
                        }
                    )
            else:
                raise OpenAIChatCompletionException(
                    "Unsupported response from OpenAI api"
                )
        return response
