from typing import Union

from openai import AzureOpenAI, OpenAI

from actionweaver.llms.azure.chat_loop import ChatCompletion
from actionweaver.llms.openai.tools.chat_loop import OpenAIChatCompletion


class LLMClientWrapper:
    def __init__(self, client: Union[OpenAI, AzureOpenAI]):

        self.client = client
        if type(client) == OpenAI:
            self.chat_loop = OpenAIChatCompletion.wrap_chat_completion_create(
                client.chat.completions.create
            )
        elif type(client) == AzureOpenAI:
            self.chat_loop = ChatCompletion.wrap_chat_completion_create(
                client.chat.completions.create
            )
        else:
            raise NotImplementedError(f"Client type {type(client)} is not supported.")

    def create(self, *args, **kwargs):
        return self.chat_loop(*args, **kwargs)


def wrap(client: Union[OpenAI, AzureOpenAI]):
    return LLMClientWrapper(client)
