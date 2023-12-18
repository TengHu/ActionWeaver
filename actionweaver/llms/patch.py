from typing import Union

from openai import AsyncOpenAI, OpenAI

from actionweaver.llms.openai.tools.chat import OpenAIChatCompletion

# from actionweaver.llms.openai.azure.chat import OpenAIChatCompletion


def patch(client: Union[OpenAI, AsyncOpenAI]):
    if isinstance(client, AsyncOpenAI):
        raise NotImplementedError(
            "AsyncOpenAI client is not supported for patching yet."
        )
    elif isinstance(client, OpenAI):
        return OpenAIChatCompletion.patch(client)
    else:
        raise TypeError(
            f"Client type {type(client)} is not supported for patching yet."
        )
