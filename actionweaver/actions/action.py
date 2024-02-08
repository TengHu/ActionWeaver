from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from openai import AzureOpenAI, OpenAI

from actionweaver.telemetry import traceable
from actionweaver.utils import DEFAULT_ACTION_SCOPE


class ActionException(Exception):
    pass


class Action:
    def __init__(
        self,
        name,
        function,
        pydantic_model,
        stop=False,  # TODO: move all `stop` argument, self.orch in chat.completions.create
        decorators: List[Callable[..., None]] = [],
        description=None,
        logger=None,
        logging_metadata: Optional[dict] = None,
        logging_level=logging.INFO,
    ):
        self.name = name
        self.logger = logger
        self.stop = stop
        self.decorators = decorators

        if function.__doc__ is None and description is None:
            raise ActionException(
                f"Decorated method under action {name} must have a docstring for description."
            )
        self.description = description or function.__doc__

        self.pydantic_model = pydantic_model

        self.undecorated_function = function
        for decorator in self.decorators:
            function = decorator(function)
        self.function = function

        if self.logger:
            self.function = traceable(
                self.name,
                self.logger,
                metadata=logging_metadata,
                level=logging_level,
            )(self.function)

        self.__module__ = self.function.__module__
        self.__name__ = self.function.__name__
        self.__qualname__ = self.function.__qualname__
        self.__annotations__ = self.function.__annotations__
        self.__doc__ = self.function.__doc__

    def json_schema(self):
        return self.pydantic_model.model_json_schema()

    def invoke(
        self,
        client=None,
        force=True,
        logger=None,
        token_usage_tracker=None,
        *args,
        **kwargs,
    ):
        from actionweaver.llms.wrapper import ActionWeaverLLMClientWrapper

        if type(client) in (OpenAI, AzureOpenAI):
            return client.chat.completions.create(
                actions=[self],
                orch={DEFAULT_ACTION_SCOPE: self, self.name: None} if force else None,
                logger=logger,
                token_usage_tracker=token_usage_tracker,
                *args,
                **kwargs,
            )
        elif type(client) == ActionWeaverLLMClientWrapper:
            return client.create(
                actions=[self],
                orch={DEFAULT_ACTION_SCOPE: self, self.name: None} if force else None,
                logger=logger,
                token_usage_tracker=token_usage_tracker,
                *args,
                **kwargs,
            )
        else:
            raise ActionException(
                f"Client type {type(client)} not supported in invoke method. Please use OpenAI or AzureOpenAI client."
            )

    def get_function_details(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema(),
        }

    def bind(self, instance) -> InstanceAction:
        return InstanceAction(
            self.name,
            self.function,
            self.pydantic_model,
            self.logger,
            self.stop,
            instance=instance,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        response = self.function(*args, **kwargs)

        return response

    def __get__(self, instance, owner) -> InstanceAction:
        """

        Note:
            The `__get__` method is a descriptor method that is called when the action is accessed from an instance.
            It returns an instance-specific action method that is bound to the given instance.
        """
        return InstanceAction(
            self.name,
            self.function,
            self.pydantic_model,
            self.logger,
            self.stop,
            instance=instance,
        )

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __eq__(self, other):
        return isinstance(other, Action) and self.name == other.name

    def __str__(self):
        return self.name


class InstanceAction(Action):
    def __init__(
        self,
        name,
        function,
        pydantic_model,
        logger=None,
        stop=False,
        instance=None,
    ):
        super().__init__(name, function, pydantic_model, stop=stop, logger=logger)
        self.instance = instance
        self.pydantic_model = pydantic_model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        response = self.function(self.instance, *args, **kwargs)

        return response


class ActionHandlers:
    def __init__(self, *args, **kwargs):
        self.name_to_action: Dict[str, Action] = {}

    def contains(self, name) -> bool:
        return name in self.name_to_action

    @classmethod
    def from_actions(cls, actions):
        ret = cls()
        ret.name_to_action = {action.name: action for action in actions}
        return ret

    def __len__(self) -> int:
        return len(self.name_to_action)

    def check_orchestration_expr_validity(self, expr):
        if expr is None:
            return

        if isinstance(expr, str):
            if expr not in self.name_to_action:
                raise ActionException(f"Action {expr} not found.")
            return

        for element in expr:
            self.check_orchestration_expr_validity(element)

    @classmethod
    def merge(cls, *handlers) -> ActionHandlers:
        merged = cls()
        for handler in handlers:
            merged.name_to_action.update(handler.name_to_action)
        return merged

    def __getitem__(self, key) -> Action:
        return self.name_to_action[key]
