# TODO: assume all actions are functions for now
from actionweaver.actions import Action


class ToolException(Exception):
    pass


class Tools:
    def __init__(self, tools=None) -> None:
        self.tools = tools

    @classmethod
    def from_expr(cls, expr):
        if expr is None:
            return cls()
        elif isinstance(expr, list):
            return cls(
                tools=expr,
            )
        else:
            raise ToolException(f"Invalid orchestration expression: {expr}")

    def to_arguments(self):
        return "\n".join(
            [
                f"""{a.name}: \n description: {a.description} \n params: {a.json_schema()}"""
                for a in self.tools
            ]
        )
