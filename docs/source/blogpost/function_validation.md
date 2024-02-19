# Enhancing Robustness: Using Pydantic Validation Decorator with ActionWeaver

In this post, we are going to showcase how we combine Pydantic validation decorator with ActionWeaver to create robust and self healing LLM function-calling applications.

- [ActionWeaver](https://github.com/TengHu/ActionWeaver) simplifies the development of LLM applications by providing straightforward tools for structured data parsing, function dispatching and orchestration.
- [Pydantic](https://docs.pydantic.dev/latest/) is the most widely used data validation library for Python. Its [@validate_call decorator](https://docs.pydantic.dev/latest/concepts/validation_decorator/) enables the validation of function arguments based on the function's annotations before the function is executed.

By the end of this blog post, you will learn how to use Pydantic validation decorator with ActionWeaver to make your function calling more robust. For the complete working notebook, please refer to [here](https://actionweaver.readthedocs.io/en/latest/notebooks/cookbooks/function_validation_and_exception_handling.html).

## Create OpenAI client, wrap with ActionWeaver.

ActionWeaver wrapper will manage the function calling loop, which includes passing function descriptions, executing functions, and handling exceptions.

```python
from openai import OpenAI
from actionweaver.llms import wrap

llm = wrap(OpenAI())
```

## Define Pydantic models with field validation
Next, let's define some Pydantic models used in the function calling. Within the Pydantic model `UserModel`, we've specified certain `field_validator`s to ensure that both the `name` and `phone_number` adhere to specific formats.

```python
import re
from pydantic import BaseModel, ValidationError, validator, PrivateAttr

class UserModel(BaseModel):
    _uid: UUID = PrivateAttr(default_factory=uuid4)
    _created_at: datetime = PrivateAttr(default_factory=datetime.now)
    name: str
    phone_number: str

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        # Split the input string into first and last names
        names = v.split()
        
        # Check if both first and last names are present
        if len(names) != 2:
            raise ValueError('Name must contain a first name and a last name separated by a space')
        
        # Check if the formatted name is not in uppercase
        if v != v.upper():
            raise ValueError('Name must be in uppercase')
        
        return v

    @field_validator('phone_number')
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        # Define a regular expression pattern for a phone number with country code
        pattern = r'^\+\d{1,3}\s*\(\d{3}\)\s*\d{3}-\d{4}$'  # Example: +1 (XXX) XXX-XXXX

        # Check if the phone number matches the pattern
        if not re.match(pattern, v):
            raise ValueError('phone number must be in the format +1 (XXX) XXX-XXXX')
        return v
```
## Create an Action from function
Now, let's use ActionWeaver `action` decorator to create an action `ingest_user_info`, which accepts list of Pydantic model `UserModel` as arguments. 

> **Note**: An `action` represents a tool that can be used by LLM. Each action comprises two main elements: a Pydantic model that is generated to facilitate structured prompting, and a conventional Python function. ActionWeaver will treat the function docstring as function description. Please refer to the [ActionWeaver documentation](https://github.com/TengHu/ActionWeaver) for more information and specifics.

We want the LLM to extract user information from natural language and invoke `ingest_user_info`, which will then store the validated user information into `user_db`.

**A [`validate_call` decorator](https://docs.pydantic.dev/latest/concepts/validation_decorator/) is added to apply validation to the arguments.**

```python
from pydantic import validate_call

user_db = []

@action(name="SaveUserInfo", stop=True)
@validate_call
def ingest_user_info(users: List[UserModel]):
    """Save user info to database"""
    user_db.append(users)
    return "success"
```

## Handling Exception

Now what if the LLM invoke the function with invalid arguments? 

Fortunately, ActionWeaver offers an ExceptionHandler feature. This handler allows users to define how to respond to exceptions within the function calling loop. 

For example, we can create a handler that takes the exception message as input for the LLM and permits a maximum number of retries. This setup enables the LLM to respond to error messages effectively.

```python
from actionweaver.llms import ExceptionHandler, ExceptionAction, ChatLoopInfo, Continue, Return

class ExceptionRetryHandler(ExceptionHandler):
    def __init__(self, max_retry=2):
        self.max_retry = max_retry

    def handle_exception(self, e: Exception, info: ChatLoopInfo) -> ExceptionAction:
        if self.max_retry:
            self.max_retry -= 1
            
            print(f"\nRetrying. Retries left: {self.max_retry}")
            print(f"Exception raised: {type(e).__name__}: {str(e)}")
            
            response = info.context['response']
            messages = info.context['messages']
            messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": response.choices[0].message.tool_calls[0].id,
                        "name": response.choices[0].message.tool_calls[0].function.name,
                        "content": f"Exceptions raised: \n{e}",
                    }
                )

            return Continue(functions=info.context['tools'])
        raise e
```

## Demo

Now, let's attempt to prompt the LLM to call `ingest_user_info`. As you'll notice, the input text below has a different format and may not pass the field validation.

> **Note**: For details on how to force the LLM to execute function, please refer to [ActionWeaver Document](https://github.com/TengHu/ActionWeaver?tab=readme-ov-file#force-execution-of-an-action).

```python
input = """                Name       Phone Number
0  Dr. Danielle King      (844)055-3780
1        John Miller  +1-268-920-5475x5
2    Michael Johnson  +1-758-232-6153x8
"""

messages = [
    {"role": "user", "content": input}
]

response = ingest_user_info.invoke(
    llm, 
    messages=messages,
    model=MODEL,
    stream=False, 
    temperature=1,
    exception_handler = ExceptionRetryHandler(3)
)
```
We anticipate the LLM to call the `ingest_user_info` function, resulting in the formatted data being inserted into the `user_db` as depicted below.

```python
[[UserModel(name='DANIELLE KING', phone_number='+1 (844) 055-3780'),
  UserModel(name='JOHN MILLER', phone_number='+1 (268) 920-5475'),
  UserModel(name='MICHAEL JOHNSON', phone_number='+1 (758) 232-6153')]]
```

For the complete working notebook, please refer to [here](https://actionweaver.readthedocs.io/en/latest/notebooks/cookbooks/function_validation_and_exception_handling.html).