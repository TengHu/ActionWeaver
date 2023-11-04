import types

from pydantic import BaseModel

from actionweaver.actions.action import Action, action


def from_model(
    model: BaseModel, stop=True, name: str = None, description: str = None
) -> Action:
    def return_kwargs_values(**kwargs):
        (value,) = kwargs.values()  # assert there is a tuple with only one element
        return value

    if name is None:
        name = f"Create{model.__name__}"

    if description is None:
        description = f"Extract {model.__name__}"

    return_kwargs_values.__doc__ = description

    return action(name=name, stop=stop)(return_kwargs_values).build_pydantic_model_cls(
        override_params={model.__name__.lower(): (model, ...)}
    )
