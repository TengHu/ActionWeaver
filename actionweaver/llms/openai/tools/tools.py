from actionweaver.actions.orchestration_expr import (
    _ActionDefault,
    _ActionHandlerLLMInvoke,
    _ActionHandlerRequired,
    _ActionHandlerSelectOne,
)

# TODO: assume all actions are functions for now


class ToolException(Exception):
    pass


class Tools:
    def __init__(self, tool_choice=None, tools=None) -> None:
        self.tools = tools
        self.tool_choice = tool_choice

    @classmethod
    def from_expr(cls, expr, action_handlers):
        if isinstance(expr, (_ActionHandlerLLMInvoke, _ActionDefault)):
            return cls()
        elif isinstance(expr, _ActionHandlerRequired):
            return cls(
                tool_choice={
                    "type": "function",
                    "function": {
                        "name": action_handlers.name_to_action[expr.action].name
                    },
                },
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": action_handlers.name_to_action[expr.action].name,
                            "description": action_handlers.name_to_action[
                                expr.action
                            ].description,
                            "parameters": action_handlers.name_to_action[
                                expr.action
                            ].json_schema(),
                        },
                    }
                ],
            )
        elif isinstance(expr, _ActionHandlerSelectOne):
            return cls(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": action_handlers.name_to_action[action].name,
                            "description": action_handlers.name_to_action[
                                action
                            ].description,
                            "parameters": action_handlers.name_to_action[
                                action
                            ].json_schema(),
                        },
                    }
                    for action in expr
                ],
                tool_choice="auto",
            )
        else:
            raise ToolException(f"Invalid orchestration expression: {expr}")

    def to_arguments(self):
        return {
            "tools": self.tools,
            "tool_choice": self.tool_choice,
        }
