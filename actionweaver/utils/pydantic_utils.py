from inspect import getfullargspec
from typing import Any, Callable, Type

from pydantic import BaseModel, create_model


def create_pydantic_model_from_func(
    func: Callable,
    model_name: str,
    base_model: Type[BaseModel] = BaseModel,
    models=None,  # models: Optional pydantic models needed for the pydantic model from function signature
    override_params=None,  # override_params: Optional dictionary of parameters to override kwarg and non-kwarg.
):
    if models is None:
        models = []

    # Retrieve function signature details
    """
    - args is a list of the positional parameter names.
    - varargs is the name of the * parameter or None if arbitrary positional arguments are not accepted.
    - varkw is the name of the ** parameter or None if arbitrary keyword arguments are not accepted.
    - defaults is an n-tuple of default argument values corresponding to the last n positional parameters, or None if there are no such defaults defined.
    - kwonlyargs is a list of keyword-only parameter names in declaration order.
    - kwonlydefaults is a dictionary mapping parameter names from kwonlyargs to the default values used if no argument is supplied.
    - annotations is a dictionary mapping parameter names to annotations.
    """
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

    """ Create a dictionary for all arguments with their annotations and defaults
    e.g. 
    {
        'name': ('str', 'Default Name'),
        'age': ('int', 0),
        'email': ('str', None),
    }
    """

    params = {}

    if override_params is None:
        models_dict = {model.__name__: model for model in models}
        for param, default in zip(args, defaults):
            annotation = annotations.get(param, Any)
            annotation = models_dict.get(annotation, annotation)
            params[param] = (annotation, default)
    else:
        # use override_params instead
        params = override_params
        keyword_only_params = {}

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
