from __future__ import annotations

import logging
from typing import Any, Dict

from actionweaver.utils import create_pydantic_model_from_func


class ActionException(Exception):
    pass


def action(name, scope="global", logger=None, models=[]):
    """
    Decorator function to create an Action object.

    Args:
    - name (str): Name of the action.
    - scope (str): Scope of the action, default is "global".
    - logger (logging.Logger): Logger instance to log information, default is None.
    - models (list[pydantic.BaseModel]): List of pydantic models to be used in the action.

    Returns:
    - create_action: A function that takes a decorated object and returns an Action object.
    """

    _logger = logger or logging.getLogger(__name__)

    def create_action(decorated_obj):
        _logger.debug({"message": f"Creating action with name: {name}, scope: {scope}"})

        action = Action(
            name=name, scope=scope, decorated_obj=decorated_obj, logger=_logger
        ).build_pydantic_model_cls(models=models)

        return action

    return create_action


class Action:
    def __init__(
        self,
        name,
        scope,
        decorated_obj,
        logger,
    ):
        self.name = name
        self.scope = scope
        self.logger = logger

        if decorated_obj.__doc__ is None:
            raise ActionException(
                f"Decorated method under action {name} must have a docstring for description."
            )

        self.description = decorated_obj.__doc__
        self.pydantic_cls = None

        self.decorated_method = decorated_obj

    def build_pydantic_model_cls(self, models=None):
        if models is None:
            models = []

        self.pydantic_cls = create_pydantic_model_from_func(
            self.decorated_method, self.decorated_method.__name__.title(), models=models
        )
        return self

    def json_schema(self):
        return self.pydantic_cls.model_json_schema()

    def bind(self, instance) -> InstanceAction:
        return InstanceAction(self, instance)

    def __get__(self, instance, owner) -> InstanceAction:
        """
        Bind a action with an instance.

        Note:
            The `__get__` method is a descriptor method that is called when the action is accessed from an instance.
            It returns an instance-specific action method that is bound to the given instance.
        """
        return InstanceAction(self, instance)


class InstanceAction:
    def __init__(self, action, instance):
        self.action = action
        self.__module__ = action.decorated_method.__module__
        self.__name__ = action.decorated_method.__name__
        self.__qualname__ = action.decorated_method.__qualname__
        self.__annotations__ = action.decorated_method.__annotations__
        self.__doc__ = action.decorated_method.__doc__
        self.instance = instance

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.action.logger.debug(
            {
                "message": f"[Action {self.action.name}, method {self.__name__}] Calling action: {self.action.name} with args: {args}"
            }
        )
        response = self.action.decorated_method(self.instance, *args, **kwargs)
        self.action.logger.debug(
            {
                "message": f"[Action {self.action.name}, method {self.__name__}] Received response: {response}"
            }
        )
        return response


class ActionHandlers:
    def __init__(self, *args, **kwargs):
        self.name_to_action: Dict[str, Action] = {}

    def contains(self, name) -> bool:
        return name in self.name_to_action

    def __len__(self) -> int:
        return len(self.name_to_action)

    def scope(self, scope):
        return {
            name: action
            for name, action in self.name_to_action.items()
            if action.scope == scope
        }

    def bind(self, instance) -> InstanceActionHandlers:
        return InstanceActionHandlers(instance, self)

    @classmethod
    def merge(cls, *handlers) -> ActionHandlers:
        merged = cls()
        for handler in handlers:
            merged.name_to_action.update(handler.name_to_action)
        return merged


class InstanceActionHandlers:
    def __init__(self, instance, action_handlers: ActionHandlers, *args, **kwargs):
        self.action_handlers = action_handlers
        self.instance = instance

    def __getitem__(self, key) -> InstanceAction:
        val = self.action_handlers.name_to_action[key]
        return val.bind(self.instance)

    def __len__(self) -> int:
        return len(self.action_handlers)

    def scope(self, scope):
        return self.action_handlers.scope(scope)

    def contains(self, name) -> bool:
        return self.action_handlers.contains(name)
