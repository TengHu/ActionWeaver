from __future__ import annotations

from tkinter import N

from actionweaver.actions.action import Action, ActionHandlers
from actionweaver.llms.openai.chat import OpenAIChatCompletion


class ActionHandlerMixinException(Exception):
    pass


class ActionHandlerMixin:
    _action_handlers = ActionHandlers()

    def __post_init__(self):
        # bind action handlers to self
        self.instance_action_handlers = self._action_handlers.bind(self)

        # build action orchestration dict
        self.instance_action_handlers.build_orchestration_dict()

        for _, attr_value in tuple(self.__dict__.items()):
            if isinstance(attr_value, OpenAIChatCompletion):
                # bind instance action handlers to llm
                attr_value._bind_action_handlers(self.instance_action_handlers)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def init_decorator(original_init):
            def new_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                if type(self) == cls:
                    self.__post_init__()

            return new_init

        def check_orchestration_expr_validity(expr):
            if expr is None:
                return

            if isinstance(expr, str):
                if expr not in cls._action_handlers.name_to_action:
                    raise ActionHandlerMixinException(
                        f"Action {expr} not found in {cls.__name__}."
                    )
                return

            for element in expr:
                check_orchestration_expr_validity(element)

        # Add __post_init__ method to the subclass.
        cls.__init__ = init_decorator(cls.__init__)

        # create action handlers from base class.
        cls._action_handlers = ActionHandlers.merge(
            *[
                base_cls._action_handlers
                for base_cls in cls.__bases__
                if hasattr(base_cls, "_action_handlers")
            ]
        )

        # add action to action handlers.
        for _, attr_value in tuple(cls.__dict__.items()):
            if isinstance(attr_value, Action):
                cls._action_handlers.name_to_action[attr_value.name] = attr_value

        # check if all action orchestration expressions are valid
        for _, action in cls._action_handlers.name_to_action.items():
            if action.orch_expr:
                if len(action.orch_expr) < 2:
                    raise ActionHandlerMixinException(
                        f"Action {action.name} must has at least two elements in its orchestration expression. The first element is the action itself. For example, SelectOne([action1, action2])."
                    )
                cls._action_handlers.check_orchestration_expr_validity(action.orch_expr)

    def action_to_pyvis_network(self):
        if self.instance_action_handlers is None:
            raise ActionHandlerMixinException(
                "Action handlers not initialized. Please call __post_init__ first."
            )

        return self.instance_action_handlers.to_pyvis_network()
