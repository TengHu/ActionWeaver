from collections import UserList
from dataclasses import dataclass

from actionweaver.utils import DEFAULT_ACTION_SCOPE

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

    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data


class RequireNext(UserList):
    """
    This class represents a sequence of actions. Invoking the first action prompts
    the llm to proceed with the next action in the sequence. For example, if you have
    [action1, action2, action3], invoking action1 will prompt the llm to continue
    with action2 and subsequently action3.
    """

    def __eq__(self, other):
        return type(self) == type(other) and self.data == other.data


#################################################################################################
# The following classes are used within action handlers class, users should not use them directly
#################################################################################################


@dataclass
class _ActionHandlerLLMInvoke:
    scope: str = DEFAULT_ACTION_SCOPE

    def __hash__(self):
        return hash(f"_ActionHandlerLLMInvoke[scope={self.scope}]")


@dataclass
class _ActionHandlerRequired:
    action: str


class _ActionHandlerSelectOne(UserList):
    pass


class _ActionDefault:
    """This class represents the default action"""

    pass
