import functools
import inspect


def preserve_original_signature(decorator):
    def decorated_decorator(func):
        decorated = decorator(func)
        decorated.__signature__ = inspect.signature(
            func
        )  # Preserve the original function's signature

        return decorated

    return decorated_decorator


def cache(func):
    wrapper = functools.cache(func)
    wrapper.__signature__ = inspect.signature(
        func
    )  # Preserve the original function's signature
    return wrapper


def lru_cache(maxsize=128, typed=False):
    def decorator(func):
        wrapper = functools.lru_cache(maxsize, typed)(func)
        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator
