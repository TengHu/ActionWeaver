from __future__ import annotations

import logging
from typing import Any, Dict

from actionweaver.actions import orchestration
from actionweaver.actions.orchestration import Orchestration
from actionweaver.actions.orchestration_expr import (
    RequireNext,
    SelectOne,
    _ActionHandlerLLMInvoke,
    _ActionHandlerRequired,
    _ActionHandlerSelectOne,
)
from actionweaver.utils import DEFAULT_ACTION_SCOPE
from actionweaver.utils.pydantic_utils import create_pydantic_model_from_func


class ActionException(Exception):
    pass


def action(
    name,
    scope=DEFAULT_ACTION_SCOPE,
    logger=None,
    orch_expr=None,
    models=[],
    stop=False,
):
    """
    Decorator function to create an Action object.

    Args:
    - name (str): Name of the action.
    - scope (str): Scope of the action, default is DEFAULT_ACTION_SCOPE.
    - logger (logging.Logger): Logger instance to log information, default is None.
    - models (list[pydantic.BaseModel]): List of pydantic models to be used in the action.
    - stop (bool): If True, the agent will stop immediately after invoking this action.
    Returns:
    - create_action: A function that takes a decorated object and returns an Action object.
    """

    _logger = logger or logging.getLogger(__name__)

    def create_action(decorated_obj):
        _logger.debug({"message": f"Creating action with name: {name}, scope: {scope}"})

        action = Action(
            name=name,
            scope=scope,
            decorated_obj=decorated_obj,
            orch_expr=orch_expr,
            logger=_logger,
            stop=stop,
        ).build_pydantic_model_cls(models=models)

        return action

    return create_action


class Action:
    def __init__(
        self,
        name,
        decorated_obj,
        scope=None,
        orch_expr=None,
        logger=None,
        stop=False,
    ):
        self.name = name
        self.scope = scope or DEFAULT_ACTION_SCOPE
        self.logger = logger
        self.orch_expr = orch_expr
        self.stop = stop

        if decorated_obj.__doc__ is None:
            raise ActionException(
                f"Decorated method under action {name} must have a docstring for description."
            )

        self.description = decorated_obj.__doc__
        self.pydantic_cls = None

        self.decorated_method = decorated_obj

        self.__module__ = self.decorated_method.__module__
        self.__name__ = self.decorated_method.__name__
        self.__qualname__ = self.decorated_method.__qualname__
        self.__annotations__ = self.decorated_method.__annotations__
        self.__doc__ = self.decorated_method.__doc__

    def build_pydantic_model_cls(
        self,
        models=None,
        override_params=None,  # override_params: Optional dictionary of parameters to override kwarg and non-kwarg of decorated method.
    ):
        if models is None:
            models = []

        self.pydantic_cls = create_pydantic_model_from_func(
            self.decorated_method,
            self.decorated_method.__name__.title(),
            models=models,
            override_params=override_params,
        )
        return self

    def json_schema(self):
        return self.pydantic_cls.model_json_schema()

    def invoke(self, chat, messages, force=True, stream=False):
        assert len(messages) > 0, "Messages cannot be empty"

        if messages is None:
            messages = []

        return chat.create(
            messages,
            actions=[self],
            orch_expr=RequireNext([self.name]) if force else None,
            stream=stream,
        )

    def bind(self, instance) -> InstanceAction:
        return InstanceAction(
            self.name,
            self.decorated_method,
            self.pydantic_cls,
            self.scope,
            self.orch_expr,
            self.logger,
            self.stop,
            instance=instance,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.logger:
            self.logger.debug(
                {
                    "message": f"[Action {self.name}, method {self.__name__}] Calling action: {self.name} with args: {args}"
                }
            )
        response = self.decorated_method(*args, **kwargs)
        if self.logger:
            self.logger.debug(
                {
                    "message": f"[Action {self.name}, method {self.__name__}] Received response: {response}"
                }
            )
        return response

    def __get__(self, instance, owner) -> InstanceAction:
        """

        Note:
            The `__get__` method is a descriptor method that is called when the action is accessed from an instance.
            It returns an instance-specific action method that is bound to the given instance.
        """
        return InstanceAction(
            self.name,
            self.decorated_method,
            self.pydantic_cls,
            self.scope,
            self.orch_expr,
            self.logger,
            self.stop,
            instance=instance,
        )


class InstanceAction(Action):
    def __init__(
        self,
        name,
        decorated_obj,
        pydantic_cls,
        scope=None,
        orch_expr=None,
        logger=None,
        stop=False,
        instance=None,
    ):
        super().__init__(name, decorated_obj, scope, orch_expr, logger, stop)
        self.instance = instance
        self.pydantic_cls = pydantic_cls

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.logger:
            self.logger.debug(
                {
                    "message": f"[Action {self.name}, method {self.__name__}] Calling action: {self.name} with args: {args}"
                }
            )
        response = self.decorated_method(self.instance, *args, **kwargs)
        if self.logger:
            self.logger.debug(
                {
                    "message": f"[Action {self.name}, method {self.__name__}] Received response: {response}"
                }
            )
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

    def scope(self, scope):
        return {
            name: action
            for name, action in self.name_to_action.items()
            if action.scope == scope
        }

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
