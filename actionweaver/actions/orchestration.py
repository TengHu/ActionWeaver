from collections import UserList
from dataclasses import dataclass

#################################################################################################
# The following classes are used within action decorators
#################################################################################################


class SelectOne(UserList):
    """
    This class represents a sequence of actions. When the first action is invoked,
    it prompts for a choice from the remaining actions. For instance, if you have
    a list [action1, action2, action3], invoking action1 will prompt a choice to select
    one action from action2 and action3, or not action (default) .
    """

    def __init__(self, data):
        if len(data) <= 1:
            raise ValueError("SelectOne must have more than 1 element")
        super().__init__(data)

    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data


class RequireNext(UserList):
    """
    This class represents a sequence of actions. Invoking the first action prompts
    the llm to proceed with the next action in the sequence. For example, if you have
    [action1, action2, action3], invoking action1 will prompt the llm to continue
    with action2 and subsequently action3.
    """

    def __init__(self, data):
        if len(data) <= 1:
            raise ValueError("RequireNext must have more than 1 element")
        super().__init__(data)

    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data


#################################################################################################
# The following classes are used within action handlers class
#################################################################################################


@dataclass
class _ActionHandlerLLMInvoke:
    scope: str = "global"

    def __hash__(self):
        return hash(f"_ActionHandlerLLMInvoke[scope={self.scope}]")


@dataclass
class _ActionHandlerRequired:
    action: str


class _ActionHandlerSelectOne(UserList):
    pass
