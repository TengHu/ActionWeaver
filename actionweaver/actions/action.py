from __future__ import annotations

import logging
from typing import Any, Dict

from actionweaver.actions.orchestration import (
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
        if self.action.logger:
            self.action.logger.debug(
                {
                    "message": f"[Action {self.action.name}, method {self.__name__}] Calling action: {self.action.name} with args: {args}"
                }
            )
        response = self.action.decorated_method(self.instance, *args, **kwargs)
        if self.action.logger:
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


class ActionOrchestrationParseError(Exception):
    pass


def parse_orchestration_expr(expr):
    orch_dict = {}

    def get_first_action(l):
        """
        Get first action name from list expression.
        """
        if isinstance(l, (SelectOne, RequireNext)):
            return get_first_action(l[0])
        elif isinstance(l, str):
            return l
        else:
            raise ActionOrchestrationParseError(
                f"Invalid object in orchestration: {l}."
            )

    def get_last_action(l):
        """
        Get last action name from list expression.
        """
        if isinstance(l, SelectOne):
            raise ActionOrchestrationParseError(
                f"Can't decide last action of SelectOne: {l}."
            )
        elif isinstance(l, RequireNext):
            return get_last_action(l[-1])
        elif isinstance(l, str):
            return l
        else:
            raise ActionOrchestrationParseError(
                f"Invalid object in orchestration: {l}."
            )

    def parse(l):
        """
        Parse list expression into orch_dict.
        """

        if isinstance(l, SelectOne):
            curr = get_last_action(l[0])
            parse(l[0])

            next_actions = []
            for n in list(l)[1:]:
                next_actions.append(get_first_action(n))
                parse(n)

            if curr in orch_dict:
                raise ActionOrchestrationParseError(f"Inconsistency caused by {l}")
            orch_dict[curr] = _ActionHandlerSelectOne(next_actions)
        elif isinstance(l, RequireNext):
            parse(l[0])
            prev, curr = get_last_action(l[0]), None
            for n in range(1, len(l)):
                curr = get_first_action(l[n])

                if prev in orch_dict:
                    raise ActionOrchestrationParseError(f"Inconsistency caused by {l}")
                orch_dict[prev] = _ActionHandlerRequired(get_first_action(curr))
                parse(l[n])

                prev = curr
        elif isinstance(l, str):
            return
        else:
            raise ActionOrchestrationParseError(f"Invalid orchestration expr: {l}.")

    parse(expr)
    return orch_dict


class InstanceActionHandlers:
    def __init__(self, instance, action_handlers: ActionHandlers, *args, **kwargs):
        self.action_handlers = action_handlers
        self.instance = instance
        self.orch_dict = {}

    def __getitem__(self, key) -> InstanceAction:
        val = self.action_handlers.name_to_action[key]
        return val.bind(self.instance)

    def __len__(self) -> int:
        return len(self.action_handlers)

    def scope(self, scope):
        return self.action_handlers.scope(scope)

    def contains(self, name) -> bool:
        return self.action_handlers.contains(name)

    def build_orchestration_dict(self):
        """
        Parse orchestration expressions from all actions.
        """
        orch_dict = {
            _ActionHandlerLLMInvoke(DEFAULT_ACTION_SCOPE): _ActionHandlerSelectOne([])
        }

        for _, action in self.action_handlers.name_to_action.items():
            if action.orch_expr:
                if action.orch_expr[0] != action.name:
                    raise ActionOrchestrationParseError(
                        f"Invalid expression {action.orch_expr} in Action {action.name}, "
                        "The first element of the orchestration expression must be the action name. "
                        "For example, in 'action1', 'SelectOne' should be used as 'SelectOne([action1, ...])', "
                        "and 'RequireNext' should be used as 'RequireNext([action1, ...])'."
                    )

                orch_dict.update(**parse_orchestration_expr(action.orch_expr))

            # Parse scope
            if (
                _ActionHandlerLLMInvoke(action.scope) not in orch_dict
                or orch_dict[_ActionHandlerLLMInvoke(action.scope)] is None
            ):  # if scope doesn't exist or map to None like DEFAULT_ACTION_SCOPE, then create a new one
                orch_dict[
                    _ActionHandlerLLMInvoke(action.scope)
                ] = _ActionHandlerSelectOne([])
            elif not isinstance(
                orch_dict[_ActionHandlerLLMInvoke(action.scope)],
                _ActionHandlerSelectOne,
            ):
                raise ActionOrchestrationParseError(
                    f"The scope {action.scope} of Action {action.name} causes inconsistency in orchestration."
                )
            orch_dict[_ActionHandlerLLMInvoke(action.scope)].append(action.name)

        self.orch_dict = orch_dict
        return self

    def to_pyvis_network(self):
        """Experimental for debugging purpose"""
        from pyvis.network import Network

        def _generate_node_config(n):
            if isinstance(n, _ActionHandlerLLMInvoke):
                if n.scope == DEFAULT_ACTION_SCOPE:
                    return {
                        "n_id": str(n),
                        "label": "LLM[DEFAULT]",
                        "color": "#005B00",
                        "physics": False,
                    }
                else:
                    return {
                        "n_id": str(n),
                        "label": f"LLM[{n.scope}]",
                        "color": "#00CC99",
                        "physics": False,
                    }
            elif isinstance(n, str):
                return {
                    "n_id": n,
                    "label": None,
                    "color": "#FF7769",
                    "physics": False,
                }
            else:
                raise ActionException("unsupported node type for visualization")

        g = Network(notebook=True, directed=True)

        for node, expr in self.orch_dict.items():
            g.add_node(**_generate_node_config(node))

            if isinstance(expr, _ActionHandlerSelectOne):
                for dest in expr:
                    g.add_node(**_generate_node_config(dest))
                    g.add_edge(
                        _generate_node_config(node)["n_id"],
                        _generate_node_config(dest)["n_id"],
                        color="#00CC99",
                        label="select",
                        physics=False,
                    ),
            elif isinstance(expr, _ActionHandlerRequired):
                g.add_node(**_generate_node_config(expr.action))
                g.add_edge(
                    _generate_node_config(node)["n_id"],
                    _generate_node_config(expr.action)["n_id"],
                    color="#FF0000",
                    label="requires",
                    physics=False,
                ),
        return g
