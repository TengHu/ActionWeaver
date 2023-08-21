import json
import logging

import openai

from actionweaver.action import ActionHandlers
from actionweaver.llms.openai.tokens import TokenUsageTracker


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
        if scope is None:
            scope = "global"

        self.logger.debug(
            f"[OpenAIChatCompletion {self.model}] Creating chat completion with scope: {scope} and messages: {messages}"
        )

        functions = []
        if len(self.action_handlers) > 0:
            functions = [
                {
                    "name": action.name,
                    "description": action.description,
                    "parameters": action.json_schema(),
                }
                for _, action in self.action_handlers.scope(scope).items()
            ]

        response = None
        # Action dispatch loop
        while True:
            self.logger.debug(
                f"[OpenAIChatCompletion {self.model}] Calling openai with messages: {messages}. "
                f"Available functions for the given scope {scope}: {[func['name'] for func in functions]}"
            )

            if functions is not None and len(functions) > 0:
                api_response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    functions=functions,
                    *args,
                    **kwargs,
                )

            else:
                api_response = openai.ChatCompletion.create(
                    model=self.model, messages=messages, *args, **kwargs
                )

            self.token_usage_tracker.track_usage(api_response.usage)

            self.logger.debug(
                f"[OpenAIChatCompletion {self.model}] Received response from OpenAI api : {api_response}"
            )

            message = api_response.choices[0]["message"]
            if "content" in message and message["content"]:
                response = message["content"]
                messages += [{"role": "assistant", "content": message["content"]}]
                self.logger.debug(
                    f"[OpenAIChatCompletion {self.model}] Processing message response: {message}"
                )

            if "function_call" not in message:
                break

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
                f"[OpenAIChatCompletion {self.model}] Processing openai function call: {name} with arguments: {arguments}"
            )

            if self.action_handlers.contains(name):
                try:
                    arguments = json.loads(message["function_call"]["arguments"])
                except json.decoder.JSONDecodeError as e:
                    self.logger.error(
                        f"[OpenAIChatCompletion {self.model}] Parsing function call arguments from openai response :{e}"
                    )
                    raise e

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
                    f"[OpenAIChatCompletion {self.model}] Action: {name} invoked with arguments: {arguments}. \n\nResponse: {function_response}"
                )

            else:
                content = f"{name} is not a valid function name, use one of the following: {', '.join([func['name'] for func in functions])}"
                messages += [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
                self.logger.debug(
                    f"[OpenAIChatCompletion {self.model}] Unavailable action: {name} with arguments: {arguments}. \n\nResponse: {content}"
                )

        return response
