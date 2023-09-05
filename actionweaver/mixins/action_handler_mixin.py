from __future__ import annotations

from tkinter import N

from actionweaver.actions.action import Action, ActionHandlers
from actionweaver.llms.openai import chat
from actionweaver.llms.openai.chat import OpenAIChatCompletion


class ActionHandlerMixinException(Exception):
    pass


class ActionHandlerMixin:
    _action_handlers = ActionHandlers()

    def __post_init__(self):
        # check if all action orchestration expressions are valid
        for _, action in self._action_handlers.name_to_action.items():
            if action.orch_expr:
                if len(action.orch_expr) < 2:
                    raise ActionHandlerMixinException(
                        f"Action {action.name} must has at least two elements in its orchestration expression. The first element is the action itself. For example, SelectOne([{action.name}, action1, action2])."
                    )
                self._action_handlers.check_orchestration_expr_validity(
                    action.orch_expr
                )

        # bind action handlers to self
        self.instance_action_handlers = self._action_handlers.bind(self)

        # build action orchestration dict
        self.instance_action_handlers.build_orchestration_dict()

        chat_completion_found = False
        for _, attr_value in tuple(self.__dict__.items()):
            if isinstance(attr_value, OpenAIChatCompletion):
                if chat_completion_found:
                    raise ActionHandlerMixinException(
                        "Only one OpenAIChatCompletion instance is allowed in a class."
                    )

                # bind instance action handlers to llm
                attr_value._bind_action_handlers(self.instance_action_handlers)
                chat_completion_found = True

        if not chat_completion_found:
            raise ActionHandlerMixinException(
                "An OpenAIChatCompletion instance is required in a class."
            )

    @classmethod
    def __post_init_subclass__(cls, **kwargs):
        # add action to action handlers.
        for _, attr_value in tuple(cls.__dict__.items()):
            if isinstance(attr_value, Action):
                cls._action_handlers.name_to_action[attr_value.name] = attr_value

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def init_decorator(original_init):
            def new_init(self, *args, **kwargs):
                try:
                    original_init(self, *args, **kwargs)
                except Exception as e:
                    error_message = (
                        f"Error occurred while initializing the object: {e}."
                    )
                    raise ActionHandlerMixinException(error_message) from e

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

        cls.__post_init_subclass__(kwargs=kwargs)

    def action_to_pyvis_network(self):
        if self.instance_action_handlers is None:
            raise ActionHandlerMixinException(
                "Action handlers not initialized. Please call __post_init__ first."
            )

        return self.instance_action_handlers.to_pyvis_network()
