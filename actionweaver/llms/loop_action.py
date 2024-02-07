from typing import Any

from pydantic import BaseModel


class LoopAction(BaseModel):
    pass


class ReturnRightAway(LoopAction):
    content: Any


class Unknown(LoopAction):
    pass


class Continue(LoopAction):
    functions: Any
