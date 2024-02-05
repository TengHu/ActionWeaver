from typing import Callable, List

from actionweaver import Action
from actionweaver.utils.pydantic_utils import create_pydantic_model_from_func


def action_from_tool(
    tool,
    name=None,
    description=None,
    decorators: List[Callable[..., None]] = [],
    stop=False,
):
    name = name or tool.name
    description = description or tool.description

    act = Action(
        name=name,
        function=tool._run,
        pydantic_model=create_pydantic_model_from_func(
            tool.name.upper(), tool._run, ignored_params=["run_manager"]
        ),
        stop=stop,
        decorators=decorators,
        description=description,
    )

    return act
