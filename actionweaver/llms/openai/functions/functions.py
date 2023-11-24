from actionweaver.actions.orchestration_expr import (
    _ActionDefault,
    _ActionHandlerLLMInvoke,
    _ActionHandlerRequired,
    _ActionHandlerSelectOne,
)


class FunctionException(Exception):
    pass


class Functions:
    def __init__(self, function_call=None, functions=None) -> None:
        self.functions = functions
        self.function_call = function_call

    @classmethod
    def from_expr(cls, expr, action_handlers):
        if isinstance(expr, (_ActionHandlerLLMInvoke, _ActionDefault)):
            return cls()
        elif isinstance(expr, _ActionHandlerRequired):
            return cls(
                function_call={"name": expr.action},
                functions=[
                    {
                        "name": action_handlers.name_to_action[expr.action].name,
                        "description": action_handlers.name_to_action[
                            expr.action
                        ].description,
                        "parameters": action_handlers.name_to_action[
                            expr.action
                        ].json_schema(),
                    }
                ],
            )
        elif isinstance(expr, _ActionHandlerSelectOne):
            return cls(
                functions=[
                    {
                        "name": action_handlers.name_to_action[action].name,
                        "description": action_handlers.name_to_action[
                            action
                        ].description,
                        "parameters": action_handlers.name_to_action[
                            action
                        ].json_schema(),
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
