# TODO: assume all actions are functions for now
from actionweaver.actions import Action


class ToolException(Exception):
    pass


class Tools:
    def __init__(self, tool_choice=None, tools=None) -> None:
        self.tools = tools
        self.tool_choice = tool_choice

    @classmethod
    def from_expr(cls, expr):

        if expr is None:
            return cls()
        elif isinstance(expr, Action):
            return cls(
                tool_choice={
                    "type": "function",
                    "function": {"name": expr.name},
                },
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": expr.name,
                            "description": expr.description,
                            "parameters": expr.json_schema(),
                        },
                    }
                ],
            )
        elif isinstance(expr, list):
            return cls(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "description": action.description,
                            "parameters": action.json_schema(),
                        },
                    }
                    for action in expr
                ],
                tool_choice="auto",
            )
        else:
            raise ToolException(f"Invalid orchestration expression: {expr}")

    @staticmethod
    def from_action_to_json(action: Action):
        return {"type": "function", "function": action.get_function_details()}

    def to_arguments(self):
        return {
            "tools": self.tools,
            "tool_choice": self.tool_choice,
        }

    def __bool__(self):
        return bool(self.tools)
