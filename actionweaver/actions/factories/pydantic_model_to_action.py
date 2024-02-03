import types
import uuid
from typing import Callable, List

from pydantic import BaseModel

from actionweaver.actions import Action
from actionweaver.actions.factories.function import action


def truncated_uuid4():
    return str(uuid.uuid4())[:8]


# def action_from_model_v0(
#     model: BaseModel,
#     stop=True,
#     name: str = None,
#     description: str = None,
#     decorators: List[Callable[..., None]] = [],
# ) -> Action:
#     def func(*args, **kwargs):
#         if args or len(kwargs) > 1:
#             raise ValueError(
#                 f"Invalid input: The method should not have positional arguments and should only accept one keyword argument: {model.__name__.lower()} of type {model.__name__}",
#                 f"args: {args}",
#                 f"kwargs: {kwargs}",
#             )

#         if kwargs:
#             (key,) = kwargs.keys()

#             # the only keyword argument should be the model name
#             if key != model.__name__.lower():
#                 raise ValueError(
#                     f"Invalid input: The method should accept a single keyword argument: {model.__name__.lower()}",
#                     f"instead: {key}",
#                 )

#             (value,) = kwargs.values()

#             try:
#                 ret = model(**value)
#                 return ret
#             except Exception as e:
#                 raise ValueError(
#                     f"Failed to parse with the {model.__name__.lower()} model",
#                     f"value: {value}",
#                 )

#     if name is None:
#         name = f"Create{model.__name__}"

#     if description is None:
#         description = f"Extract {model.__name__}"

#     func.__doc__ = description
#     func.__name__ = f"create_{model.__name__.lower()}_from_pydantic_model"

#     return action(
#         name=name,
#         pydantic_model=create_pydantic_model_from_function(
#             func, override_params={model.__name__.lower(): (model, ...)}
#         ),
#         stop=stop,
#         decorators=decorators,
#     )(func)


def action_from_model(
    model: BaseModel,
    stop=True,
    name: str = None,
    description: str = None,
    decorators: List[Callable[..., None]] = [],
) -> Action:
    def func(*args, **kwargs):
        if args:
            raise ValueError(
                f"Invalid input: The method should not have positional arguments and should only accept keyword arguments: {model.__name__.lower()} of type {model.__name__}",
                f"args: {args}",
                f"kwargs: {kwargs}",
            )

        return model.model_validate(kwargs)

    if name is None:
        name = f"Create{model.__name__}"

    if description is None:
        description = f"Extract {model.__name__}"

    func.__doc__ = description
    func.__name__ = f"create_{model.__name__.lower()}_from_pydantic_model"

    return action(
        name=name,
        pydantic_model=model,
        stop=stop,
        decorators=decorators,
    )(func)
