import collections
from typing import List

from actionweaver.actions.action import Action, action


def combine(
    acts: List[Action], name: str = None, description: str = None, reducer=None
) -> Action:
    # TODO: make action has a default of None, so the mode can choose to use it or not
    if reducer is None:
        reducer = lambda l: "\n".join([str(e) for e in l])

    def func(*args, **kwargs):
        value_list = list(kwargs.values())
        if args:
            raise ValueError(
                "Invalid input: The method should not have positional arguments",
                f"args: {args}",
            )

        ans = []
        for a, i in zip(acts, value_list):
            try:
                ans += [a(**i)]
            except Exception as e:
                raise ValueError(
                    f"Failed to invoke {a.name} with the value of {i}: {e}",
                )

        return reducer(ans)

    # Build the name string using all action names from acts
    action_names = "_".join(a.name.lower() for a in acts)
    if name is None:
        name = f"combine_{action_names}"

    if description is None:
        description = ""

    func.__doc__ = description
    func.__name__ = name

    params = collections.OrderedDict()
    for act in acts:
        params[act.name] = (act.pydantic_cls, ...)

    return action(name=name.title())(func).build_pydantic_model_cls(
        override_params=params
    )
