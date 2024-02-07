from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel

from actionweaver.llms.loop_action import Continue as _Continue
from actionweaver.llms.loop_action import LoopAction
from actionweaver.llms.loop_action import ReturnRightAway as _ReturnRightAway
from actionweaver.llms.loop_action import Unknown as _Unknown


class ChatLoopInfo(BaseModel):
    context: Dict[str, Any]


class ExceptionAction(LoopAction):
    pass


class Continue(ExceptionAction, _Continue):
    pass


class Return(ExceptionAction, _ReturnRightAway):
    pass


class Unknown(ExceptionAction, _Unknown):
    pass


class ExceptionHandler(ABC):
    """Base class for exception handlers.

    This class provides a framework for handling exceptions within the function calling loop.
    """

    @abstractmethod
    def handle_exception(self, e: Exception, info: ChatLoopInfo) -> ExceptionAction:
        pass
