import inspect
import typing
from inspect import getfullargspec
from typing import Any, Callable, Dict, List, Type

from pydantic import BaseModel, create_model
from pydantic.config import ConfigDict


def create_pydantic_model_from_func_v0(
    func: Callable,
    model_name: str,
    base_model: Type[BaseModel] = BaseModel,
    config: ConfigDict = None,
    validators: dict[str, classmethod] = None,
    nested_models=None,  # models: Optional pydantic models needed for the pydantic model from function signature
    override_params=None,  # override_params: Optional dictionary of parameters to override kwarg and non-kwarg.
    ignored_params=None,  # ignored_params: Optional list of parameters to ignore.
):
    """
    This function is inspired by https://github.com/pydantic/pydantic/issues/1391


    Documentation:
    -------------
    1. https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation

    """
    if nested_models is None:
        nested_models = []

    """
    Retrieve function signature details
    
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
        models_dict = {model.__name__: model for model in nested_models}
        for param, default in zip(args, defaults):
            annotation = annotations.get(param, Any)
            annotation = models_dict.get(annotation, annotation)
            params[param] = (annotation, default)
    else:
        # use override_params instead
        params = override_params
        keyword_only_params = {}

    ## Remove ignored params
    if ignored_params:
        params = {
            key: value for key, value in params.items() if key not in ignored_params
        }
        keyword_only_params = {
            key: value
            for key, value in keyword_only_params.items()
            if key not in ignored_params
        }

    # # Configure the class to allow extra parameters if **kwargs is in the function signature
    # class Config:
    #     extra = "allow"

    if config:
        return create_model(
            model_name,
            **params,
            **keyword_only_params,
            __config__=config,
            __validators__=validators,
        )
    else:
        return create_model(
            model_name,
            **params,
            **keyword_only_params,
            __base__=base_model,
            __validators__=validators,
        )


def convert_default(val):
    default_mapping = {inspect._empty: Ellipsis}
    return default_mapping.get(val, val)


def convert_annotation(val):
    annotation_mapping = {inspect._empty: Any}
    return annotation_mapping.get(val, val)


def create_pydantic_model_from_func(
    model_name: str,
    func: Callable,
    base_model: Type[BaseModel] = BaseModel,
    config: ConfigDict = None,
    validators: Dict[str, classmethod] = None,
    override_params: Dict[str, Any] = None,
    ignored_params: List[str] = None,
):
    """
    Create a Pydantic model from a function signature.

    If inspect signature return string for imported methods, consider removing `from __future__ import annotations`
    Documentation:
    -------------
    1. https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation
    2. https://github.com/pydantic/pydantic/issues/1391
    """

    # Retrieve function signature details using inspect.signature
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())

    # Filter out 'self' if present
    params = {param.name: param for param in parameters if param.name != "self"}

    # Convert to a dictionary of parameter names and their annotations and defaults
    params = {
        name: (convert_annotation(param.annotation), convert_default(param.default))
        for name, param in params.items()
    }

    # Use override_params instead
    if override_params:
        params = override_params

    # Remove ignored params
    if ignored_params:
        params = {
            key: value for key, value in params.items() if key not in ignored_params
        }

    # (Deprecated) Convert annotations to pydantic models if needed
    # if supporting_annotation_models:
    #     models_dict = {model.__name__: model for model in supporting_annotation_models}
    #     for name, (annotation, default) in params.items():
    #         annotation = models_dict.get(annotation, annotation)
    #         params[name] = (annotation, default)

    if config:
        return create_model(
            model_name,
            **params,
            __config__=config,
            __validators__=validators,
        )
    else:
        return create_model(
            model_name,
            **params,
            __base__=base_model,
            __validators__=validators,
        )
