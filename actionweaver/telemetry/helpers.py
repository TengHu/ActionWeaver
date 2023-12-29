import contextvars
import functools
import inspect
import logging
import time
import traceback
import uuid
from typing import Any, Callable, Dict, Optional

_PARENT_RUN_ID = contextvars.ContextVar("_PARENT_RUN_ID", default=None)


def get_parent_run_id():
    return _PARENT_RUN_ID.get()


def _get_inputs(
    signature: inspect.Signature, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Return a dictionary of inputs from the function signature."""
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop("self", None)
    arguments.pop("cls", None)
    for param_name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # Update with the **kwargs, and remove the original entry
            # This is to help flatten out keyword arguments
            if param_name in arguments:
                arguments.update(arguments[param_name])
                arguments.pop(param_name)

    return arguments


# inspired by langsmith.run_helpers.traceable
def traceable(
    name,
    logger,
    metadata: Optional[Dict] = None,
    level=logging.INFO,
) -> Callable:
    original_metadata = metadata or {}

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(
            *args: Any,
            logging_extra: Optional[Dict] = None,
            **kwargs: Any,
        ) -> Any:
            parent_run_id = _PARENT_RUN_ID.get()

            run_id = uuid.uuid4()

            logging_extra = logging_extra or {}
            metadata = original_metadata.copy()
            metadata.update(logging_extra)

            signature = inspect.signature(func)
            inputs = _get_inputs(signature, *args, **kwargs)

            _PARENT_RUN_ID.set(run_id)
            try:
                function_result = func(*args, **kwargs)
                logger.log(
                    level,
                    {
                        "name": name,
                        "inputs": inputs,
                        "outputs": function_result,
                        "parent_run_id": parent_run_id,
                        "run_id": run_id,
                        "timestamp": time.time(),
                        **metadata,
                    },
                )
            except Exception as e:
                stacktrace = traceback.format_exc()
                logger.log(
                    level,
                    {
                        "name": name,
                        "inputs": inputs,
                        "error": stacktrace,
                        "parent_run_id": parent_run_id,
                        "run_id": run_id,
                        "timestamp": time.time(),
                        **metadata,
                    },
                )
                raise e
            finally:
                _PARENT_RUN_ID.set(parent_run_id)
            return function_result

        return wrapper

    return decorator
