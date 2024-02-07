from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel

from actionweaver.llms.loop_action import LoopAction


class ChatLoopInfo(BaseModel):
    context: Dict[str, Any]


class ExceptionHandler(ABC):
    """Base class for exception handlers.

    This class provides a framework for handling exceptions within the function calling loop.
    """

    @abstractmethod
    def handle_exception(self, e: Exception, info: ChatLoopInfo) -> LoopAction:
        pass
