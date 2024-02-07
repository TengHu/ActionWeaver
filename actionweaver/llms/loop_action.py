from typing import Any, Union

from pydantic import BaseModel, Field, PrivateAttr


class LoopAction(BaseModel):
    pass


class ReturnRightAway(LoopAction):
    content: Any


class Unknown(LoopAction):
    pass


class Continue(LoopAction):
    functions: Any
