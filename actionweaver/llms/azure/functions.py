from actionweaver.actions import Action


class FunctionException(Exception):
    pass


class Functions:
    def __init__(self, function_call=None, functions=None) -> None:
        self.functions = functions
        self.function_call = function_call

    @classmethod
    def from_expr(cls, expr):
        if expr is None:
            return cls()
        elif isinstance(expr, Action):
            return cls(
                function_call={"name": expr.name},
                functions=[
                    {
                        "name": expr.name,
                        "description": expr.description,
                        "parameters": expr.json_schema(),
                    }
                ],
            )
        elif isinstance(expr, list):
            return cls(
                functions=[
                    {
                        "name": action.name,
                        "description": action.description,
                        "parameters": action.json_schema(),
                    }
                    for action in expr
                ],
                function_call="auto",
            )
        else:
            raise FunctionException(f"Invalid orchestration expression: {expr}")

    def to_arguments(self):
        return {
            "functions": self.functions,
            "function_call": self.function_call,
        }
