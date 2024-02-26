![PyPI - Version](https://img.shields.io/pypi/v/actionweaver?color=6cc644) 
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI - Downloads](https://img.shields.io/pypi/dm/actionweaver)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/TengHu/ActionWeaver/issues)





![Logo](docs/figures/actionweaver.png)


# ActionWeaver

🪡 AI application framework that makes function calling with LLM easier🪡
- **Designed for simplicity,  only relying on OpenAI and Pydantic.**
- **Supporting both OpenAI API and Azure OpenAI service!**

[Explore Our Cookbooks For Tutorials & Examples!](https://actionweaver.readthedocs.io/en/latest/notebooks/cookbooks/cookbook.html)

[Discord](https://discord.gg/fnsnBB99C2)

---

**NOTEBOOKS TO GET STARTED:**
- [QuickStart](docs/source/notebooks/cookbooks/quickstart.ipynb)
- [Using Pydantic for Structured Data Parsing and Validation](docs/source/notebooks/cookbooks/pydantic.ipynb)
- [Function Validation with Pydantic and Exception Handler](docs/source/blogpost/function_validation.md)
- [Built Traceable Action with LangSmith Tracing](docs/source/blogpost/langsmith.md)
- [Action Orchestration](docs/source/notebooks/cookbooks/orchestration.ipynb)
- [Stateful Agent](docs/source/notebooks/cookbooks/stateful_agent.ipynb)
  
---

[Star us on Github!](https://github.com/TengHu/ActionWeaver)

ActionWeaver strives to be the most reliable, user-friendly, high-speed, and cost-effective function-calling framework for AI engineers.

Features:
- **Function Calling as First-Class Citizen**: Put function-calling at the core of the framework.
- **Extensibility**: Integrate ANY Python code into your agent's toolbox with a single line of code, or combining tools from other ecosystems like LangChain or Llama Index.
- **Function Orchestration**: Build complex orchestration of function callings. including intricate hierarchies or chains.
- **Telemetry and Observability**: Easy integration with platforms like [LangSmith to build tracable application](https://github.com/TengHu/ActionWeaver/blob/main/docs/source/blogpost/langsmith.md
). Also take a look at [this link](https://actionweaver.readthedocs.io/en/latest/notebooks/cookbooks/logging.html) to see how ActionWeaver uses structured logging.

<!--
At a high level, ActionWeaver simplifies the process of creating functions, orchestrating them, and handling the invocation loop. An "action" in this context serves as an abstraction of functions or tools that users want the Language Model (LLM) to handle.

 <img src="docs/figures/function_loop.png"> -->


## Installation
You can install ActionWeaver using pip:

```python
pip install actionweaver
```
## Quickstart

Use the **LATEST** OpenAI API that supports parallel function calling !
```python
from actionweaver.llms import wrap
from openai import OpenAI

openai_client = wrap(OpenAI())
```

or using Azure OpenAI service
```python
import os
from openai import AzureOpenAI

azure_client = wrap(AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2023-10-01-preview"
))
```
The ActionWeaver wrapped client will manage the function calling loop, which includes passing function descriptions, executing functions with arguments returned by llm, and handling exceptions.

This client will expose a `create` API built upon the original `chat.completions.create` API. The enhanced `create` API will retain all original arguments and include additional parameters such as:
- `action`: providing available actions to LLM.
- `orch`: orchestrating actions throughout the function calling loop.
- `exception_handler`: an object guiding the function calling loop on how to handle exceptions.
These arguments will be demonstrated in the subsequent sections.

These additional arguments are optional, and there's always the fallback option to access the original OpenAI client via `openai_client.client`.

> **Note**: An `action` represents a tool that can be used by LLM. Each action comprises two main elements: a Pydantic model that is auto-generated to facilitate prompting, and a conventional Python function. 

### Add ANY Python function as an action to the Large Language Model.

Developers can attach **ANY** Python function as an action with a simple decorator. In the following example, we introduce action `GetCurrentTime`, and then proceed to use the OpenAI API to invoke it.

ActionWeaver utilizes the decorated method's signature and docstring as a description, passing them along to OpenAI's function API. The Action decorator is also highly adaptable and can be combined with other decorators, provided that the original signature is preserved.

```python
from actionweaver import action

@action(name="GetCurrentTime")
def get_current_time() -> str:
    """
    Use this for getting the current time in the specified time zone.
    
    :return: A string representing the current time in the specified time zone.
    """
    import datetime
    current_time = datetime.datetime.now()
    
    return f"The current time is {current_time}"

# Ask LLM what time is it
response = openai_client.create(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": "what time is it"}],
  actions = [get_current_time]
)
```
Take a look what is passing to OpenAI API
```python
get_current_weather.get_function_details()

"""
{'name': 'GetWeather',
 'description': 'Get the current weather in a given location',
 'parameters': {'properties': {'location': {'title': 'Location'},
   'unit': {'default': 'fahrenheit', 'title': 'Unit'}},
  'required': ['location'],
  'title': 'Get_Current_Weather',
  'type': 'object'}}
"""
```


### Force execution of an action
You can also compel the language model to execute the action by calling the `invoke` method of an action. Its arguments includes the ActionWeaver-wrapped client and other arguments passed to the create API.
```python 
get_current_time.invoke(openai_client, messages=[{"role": "user", "content": "what time is it"}], model="gpt-3.5-turbo", stream=False, force=True)
```

### Structured extraction
You can create a Pydantic model to define the structural data you want to extract, create an action using `action_from_model` and then force the language model to extract structured data from information in the prompt.

```python
from pydantic import BaseModel
from actionweaver.actions.factories.pydantic_model_to_action import action_from_model

class User(BaseModel):
    name: str
    age: int

action_from_model(User, stop=True).invoke(client, messages=[{"role": "user", "content": "Tom is 31 years old"}], model="gpt-3.5-turbo", stream=False, force=False)
```
> **Note**: The `stop` property of an action, with a default value of False, determines whether the function calling loop will immediately return the action's result instead of passing it to LLM if set to True.

> **Note**: You can simultaneously pass actions generated from both functions and Pydantic models.
## Orchestration of Actions

ActionWeaver enables the design of hierarchies and chains of actions by passing in `orch` argument. `orch` is a mapping from actions as keys to values including

-  a list of actions: if the key action is invoked, LLM will proceed to choose an action from the provided list, or respond with a text message.
-  an action: after key action is invoked, LLM will invoke the value action.
-  None: after key action is invoked, LLM will respond with a text message.

For example, let's say we have actions a1, a2, a3.
 
```python
client.create(
    [
        {"role": "user", "content": "message"} 
    ],
    actions=[a1, a2, a3], # First, LLM respond with either a1, a2 or a3, or text without action
    # Define the orchestration logic for actions:
    orch={
        a1.name: [a2, a3],  # If a1 is invoked, the next response will be either a2, a3 or a text response.
        a2.name: a3,      # If a2 is invoked, the next action will be a3
        a3.name: [a4]     # If a3 is invoked, the next response will be a4 or a text response.
        a4.name: None     # If a4 is invoked, the next response will guarantee to be a text message
    }
)
```

## Exception Handling

Users can provide a specific implementation of ExceptionHandler, where the `handle_exception` method is invoked upon encountering an exception. The `info` parameter encapsulates contextual details such as messages and API responses within a dictionary.

The `handle_exception` method dictates the course of action for the function calling loop, returning either:
- `Return`: providing immediate content back to the user
- `Continue`: instructing the loop to proceed.

```python
from actionweaver.llms import  ExceptionAction, ChatLoopInfo, Continue, Return

class ExceptionHandler(ABC):
    """Base class for exception handlers.

    This class provides a framework for handling exceptions within the function calling loop.
    """

    @abstractmethod
    def handle_exception(self, e: Exception, info: ChatLoopInfo) -> ExceptionAction:
        pass
```

Take a look at this [example](https://github.com/TengHu/ActionWeaver/blob/main/docs/source/blogpost/function_validation.md) for details.

## Contributing
Contributions in the form of bug fixes, new features, documentation improvements, and pull requests are VERY welcomed.

## 📔 Citation & Acknowledgements

If you find ActionWeaver useful, please consider citing the project:

```bash
@software{Teng_Hu_ActionWeaver_2024,
    author = {Teng Hu},
    license = {Apache-2.0},
    month = Aug,
    title = {ActionWeaver: Application Framework for LLMs},
    url = {https://github.com/TengHu/ActionWeaver},
    year = {2024}
}
```
