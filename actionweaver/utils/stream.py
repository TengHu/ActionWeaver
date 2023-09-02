import inspect
from itertools import tee

from openai.openai_object import OpenAIObject


def is_generator(obj):
    return inspect.isgeneratorfunction(obj) or inspect.isgenerator(obj)


def get_first_element_and_iterator(iterator):
    # Create two copies of the iterator
    iter1, iter2 = tee(iterator, 2)

    first_element = next(iter1)
    return first_element, iter2


def merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            if isinstance(value, OpenAIObject):
                value = value.to_dict()

            if type(value) == dict:
                merged_dict[key] = merge_dicts(merged_dict[key], value)
            elif type(value) == str:
                merged_dict[key] += value
            else:
                raise Exception(f"Unsupported type {type(value)}")

        else:
            merged_dict[key] = value
    return merged_dict
