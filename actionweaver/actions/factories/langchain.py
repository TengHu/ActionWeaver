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
        name, tool._run, description=description, decorators=decorators, stop=stop
    )

    # Ignore the run_manager parameter
    act.pydantic_model = create_pydantic_model_from_func(
        tool._run, name, ignored_params=["run_manager"]
    )
    return act
