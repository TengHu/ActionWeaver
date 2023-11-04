from typing import List

from actionweaver.actions.action import Action, action


def repeat(
    act: Action, name: str = None, description: str = None, reducer=None
) -> Action:
    if reducer is None:
        reducer = lambda l: "\n".join([str(e) for e in l])

    def func(*args, **kwargs):
        if args or len(kwargs) > 1:
            raise ValueError(
                f"Invalid input: The method should not have positional arguments and should only accept one keyword argument",
                f"args: {args}",
                f"kwargs: {kwargs}",
            )

        if kwargs:
            (key,) = kwargs.keys()

            if key != act.name:
                raise ValueError(
                    f"Invalid input: The method should accept a single keyword argument: {act.name}",
                    f"instead: {key}",
                )

            (value,) = kwargs.values()
            if not isinstance(value, list):
                raise ValueError(
                    f"Invalid input: The {act.name} should have a list of {act.pydantic_cls.__name__} objects",
                    f"instead: {value}",
                )

            return reducer([act(**e) for e in value])

    if name is None:
        name = act.name

    if description is None:
        description = act.description

    func.__name__ = act.__name__
    func.__doc__ = description

    return action(name=name, stop=act.stop)(func).build_pydantic_model_cls(
        override_params={act.name: (List[act.pydantic_cls], ...)}
    )
