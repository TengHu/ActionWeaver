from __future__ import annotations

from actionweaver.actions import orchestration
from actionweaver.actions.action import Action, ActionHandlers, InstanceAction
from actionweaver.llms.openai.chat import OpenAIChatCompletion


class ActionHandlerMixinException(Exception):
    pass


class ActionHandlerMixin:
    _action_handlers = ActionHandlers()

    def __post_init__(self):
        # check if all action orchestration expressions are valid
        for name, action in self._action_handlers.name_to_action.items():
            if action.orch_expr:
                if len(action.orch_expr) < 2:
                    raise ActionHandlerMixinException(
                        f"Action {action.name} must has at least two elements in its orchestration expression. The first element is the action itself. For example, SelectOne([{action.name}, action1, action2])."
                    )
                self._action_handlers.check_orchestration_expr_validity(
                    action.orch_expr
                )

            # bind action to instance
            self._action_handlers.name_to_action[name] = action.bind(self)

        self.orch = orchestration.build_orchestration_dict(self._action_handlers)

        # Implicitly bind chat completion to action handlers and orchestration.
        chat_completion_found = False
        for _, attr_value in tuple(self.__dict__.items()):
            if isinstance(attr_value, OpenAIChatCompletion):
                if chat_completion_found:
                    raise ActionHandlerMixinException(
                        "Only one OpenAIChatCompletion instance is allowed in a class."
                    )

                # bind action handlers to chat
                attr_value._bind_action_handlers(self._action_handlers)
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
