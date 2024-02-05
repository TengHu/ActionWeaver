import logging
from typing import Any, Callable, Dict, List, Optional

from actionweaver.actions.action import Action, ActionException
from actionweaver.telemetry import traceable
from actionweaver.utils import DEFAULT_ACTION_SCOPE
from actionweaver.utils.pydantic_utils import create_pydantic_model_from_func


def create_pydantic_model_from_function(
    function,
    override_params=None,  # override_params: Optional dictionary of parameters to override kwarg and non-kwarg of decorated method.
):

    return create_pydantic_model_from_func(
        function.__name__.title(),
        function,
        override_params=override_params,
    )


def action(
    name,
    pydantic_model=None,
    logger=None,
    stop=False,
    description=None,
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
                else create_pydantic_model_from_func(
                    function.__name__.title(),
                    function,
                )
            ),
            stop=stop,
            decorators=decorators,
            description=description,
            logger=_logger,
            logging_metadata=logging_metadata,
            logging_level=logging_level,
        )

    return create_action
