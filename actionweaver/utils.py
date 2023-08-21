from inspect import getfullargspec
from typing import Any, Callable, Type

from pydantic import BaseModel, create_model


def create_pydantic_model_from_signature(
    func: Callable, model_name: str, base_model: Type[BaseModel] = BaseModel
):
    # Retrieve function signature details
    (
        args,
        varargs,
        varkw,
        defaults,
        kwonlyargs,
        kwonlydefaults,
        annotations,
    ) = getfullargspec(func)

    if len(args) > 0 and args[0] == "self":
        args = args[1:]  # skip self

    defaults = defaults or []
    args = args or []

    # Calculate the number of arguments that don't have default values
    non_default_args = len(args) - len(defaults)

    # Add placeholders for non-default arguments
    defaults = (...,) * non_default_args + tuple(defaults)

    # Create a dictionary for keyword-only arguments with their defaults
    keyword_only_params = {
        param: kwonlydefaults.get(param, Any) for param in kwonlyargs
    }
    # Create a dictionary for all arguments with their annotations and defaults
    params = {
        param: (annotations.get(param, Any), default)
        for param, default in zip(args, defaults)
    }

    # Configure the class to allow extra parameters if **kwargs is in the function signature
    class Config:
        extra = "allow"

    config = Config if varkw else None

    # Create and return the pydantic model using the gathered parameters, configurations, and base model
    return create_model(
        model_name,
        **params,
        **keyword_only_params,
        __base__=base_model,
        # __config__=config,
    )
