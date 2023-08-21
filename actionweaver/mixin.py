from __future__ import annotations

import copy

from actionweaver.action import Action, ActionHandlers
from actionweaver.llms.openai.chat import OpenAIChatCompletion


class ActionHandlerMixin:
    _action_handlers = ActionHandlers()

    def __post_init__(self):
        # bind action handlers to self
        self.instance_action_handlers = self._action_handlers.bind(self)

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

        for _, attr_value in tuple(cls.__dict__.items()):
            if isinstance(attr_value, Action):
                cls._action_handlers.name_to_action[attr_value.name] = attr_value
