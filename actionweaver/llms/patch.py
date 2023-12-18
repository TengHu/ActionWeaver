from typing import Union

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from actionweaver.llms.azure.chat import ChatCompletion
from actionweaver.llms.openai.tools.chat import OpenAIChatCompletion


def patch(client: Union[OpenAI, AsyncOpenAI, AsyncAzureOpenAI, AzureOpenAI]):
    if type(client) in (AsyncAzureOpenAI, AsyncOpenAI):
        raise NotImplementedError(
            "AsyncOpenAI client is not supported for patching yet."
        )
    elif type(client) == OpenAI:
        return OpenAIChatCompletion.patch(client)
    elif type(client) == AzureOpenAI:
        return ChatCompletion.patch(client)
    else:
        raise TypeError(
            f"Client type {type(client)} is not supported for patching yet."
        )
