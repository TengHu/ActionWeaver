import types

import uuid

from pydantic import BaseModel

from actionweaver.actions.action import Action, action



def truncated_uuid4():
    return str(uuid.uuid4())[:8]


def from_model(
    model: BaseModel, stop=True, name: str = None, description: str = None
) -> Action:
    def func(*args, **kwargs):
        if args or len(kwargs) > 1:
            raise ValueError(
                f"Invalid input: The method should not have positional arguments and should only accept one keyword argument: {model.__name__.lower()} of type {model.__name__}",
                f"args: {args}",
                f"kwargs: {kwargs}",
            )

        if kwargs:
            (key,) = kwargs.keys()

            if key != model.__name__.lower():
                raise ValueError(
                    f"Invalid input: The method should accept a single keyword argument: {model.__name__.lower()}",
                    f"instead: {key}",
                )

            (value,) = kwargs.values()

            try:
                ret = model(**value)
                return ret
            except Exception as e:
                raise ValueError(
                    f"Failed to parse with the {model.__name__.lower()} model",
                    f"value: {value}",
                )


    if name is None:
        name = f"Create{model.__name__}"

    if description is None:
        description = f"Extract {model.__name__}"

    func.__doc__ = description
    func.__name__ = f"create_{model.__name__.lower()}_{truncated_uuid4()}"

    return action(name=name, stop=stop)(func).build_pydantic_model_cls(
        override_params={model.__name__.lower(): (model, ...)}
    )
