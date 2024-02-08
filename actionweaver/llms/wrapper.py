from typing import Union

from openai import AzureOpenAI, OpenAI

from actionweaver.llms.azure.chat_loop import create_chat_loop as create_chat_loop_azure
from actionweaver.llms.openai.tools.chat_loop import create_chat_loop


class ActionWeaverLLMClientWrapper:
    def __init__(self, client: Union[OpenAI, AzureOpenAI]):

        self.client = client
        if type(client) == OpenAI:
            self.chat_loop = create_chat_loop(client.chat.completions.create)
        elif type(client) == AzureOpenAI:
            self.chat_loop = create_chat_loop_azure(client.chat.completions.create)
        else:
            raise NotImplementedError(f"Client type {type(client)} is not supported.")

    def create(self, *args, **kwargs):
        return self.chat_loop(*args, **kwargs)


def wrap(client: Union[OpenAI, AzureOpenAI]):
    return ActionWeaverLLMClientWrapper(client)
