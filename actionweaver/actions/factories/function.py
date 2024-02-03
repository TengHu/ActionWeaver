from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from actionweaver.actions.action import Action, ActionException
from actionweaver.telemetry import traceable
from actionweaver.utils import DEFAULT_ACTION_SCOPE
from actionweaver.utils.pydantic_utils import create_pydantic_model_from_func


def create_pydantic_model_from_function(
    function,
    nested_models=None,
    override_params=None,  # override_params: Optional dictionary of parameters to override kwarg and non-kwarg of decorated method.
):
    if nested_models is None:
        nested_models = []

    return create_pydantic_model_from_func(
        function,
        function.__name__.title(),
        nested_models=nested_models,
        override_params=override_params,
    )


def action(
    name,
    pydantic_model=None,
    logger=None,
    nested_models=[],
    stop=False,
    decorators: List[Callable[..., None]] = [],
    logging_metadata: Optional[dict] = None,
    logging_level=logging.INFO,
):

    _logger = logger

    def create_action(function):
        return Action(
            name=name,
            function=function,
            pydantic_model=(
                pydantic_model
                if pydantic_model
                else create_pydantic_model_from_function(
                    function, nested_models=nested_models
                )
            ),
            stop=stop,
            decorators=decorators,
            logger=_logger,
            logging_metadata=logging_metadata,
            logging_level=logging_level,
        )

    return create_action
