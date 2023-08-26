from collections import UserList


class SelectOne(UserList):
    """
    This class represents a sequence of actions. When the first action is invoked,
    it prompts for a choice from the remaining actions. For instance, if you have
    a list [action1, action2, action3], invoking action1 will prompt a choice to select
    one action from action2 and action3, or not action (default) .
    """

    pass


class RequireNext(UserList):
    """
    This class represents a sequence of actions. Invoking the first action prompts
    the llm to proceed with the next action in the sequence. For example, if you have
    [action1, action2, action3], invoking action1 will prompt the llm to continue
    with action2 and subsequently action3.
    """

    pass
