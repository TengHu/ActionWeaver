# High-Level Concepts


<img src="../../../figures/scale_tools.png">


ActionWeaver helps you easily build LLM-powered agents that excel at dispatching and orchestrating external tools efficiently at scale. It currently leverages OpenAI functions behind the scenes.

In this guide focusing on high-level concepts, you will gain insights into:

The fundamental concepts and mental models that underpin the functioning of ActionWeaver, and some examples as well.

## Agent

An agent is a class that inherits from **ActionHandlerMixin** and utilizes **Action**. An agent class is required to have a single OpenAIChatCompletion object. Upon agent initialization, the ActionHandlerMixin will parse all Actions, creating an orchestration graph. When the OpenAIChatCompletion is invoked, it will utilize this action orchestration graph in an iterative process while also making calls to the OpenAI chat completion endpoint.

Example: 
```python
import logging
from typing import List
from actionweaver import ActionHandlerMixin, action
from actionweaver.llms.openai.chat import OpenAIChatCompletion
from actionweaver.llms.openai.tokens import TokenUsageTracker

logger = logging.getLogger(__name__)

class AgentV0(ActionHandlerMixin):
    def __init__(self, logger):
        self.logger = logger
        self.token_tracker = TokenUsageTracker(budget=None, logger=logger)
        self.llm = OpenAIChatCompletion("gpt-3.5-turbo", token_usage_tracker = self.token_tracker, logger=logger)
        
        self.messages = [{"role": "system", "content": "You are a resourceful assistant."}]
        self.buffer = [] 
    
    def __call__(self, text):
        self.messages += [{"role": "user", "content":text}]
        return self.llm.create(messages=self.messages)
        
    @action(name="GetCurrentTime")
    def get_current_time(self) -> str:
        """
        Use this for getting the current time in the specified time zone.
        
        :return: A string representing the current time in the specified time zone.
        """
        import datetime
        current_time = datetime.datetime.now()
        
        return f"The current time is {current_time}"
```

## Action

Action is a core concept in ActionWeaver, where each Action corresponds to a function within the OpenAI realm. In ActionWeaver, you have the capability to convert any Python function into an action that your agent can dispatch by merely adding an action decorator, as demonstrated in the example above. The agent will utilize the function's docstring as the description for the OpenAI API.

### Grouping and Extending Actions Through Inheritance

Users can also inherit actions from parent ActionWeaver agent class. In this example below, , creating a powerful and extensible design pattern. 

In the example below, we wrap the [LangChain Google search](https://python.langchain.com/docs/integrations/tools/google_search) as a method. With inheritance, the new agent can utilize the Google search tool method as well as any other actions defined in the parent classes. This structure makes it simple to compose agent upon existing code.

```python
class LangChainTools(ActionHandlerMixin):
    @action(name="GoogleSearch")
    def google_search(self, query: str) -> str:
        """
        Perform a Google search using the provided query. 
        
        This action requires `langchain` and `google-api-python-client` installed, and GOOGLE_API_KEY, GOOGLE_CSE_ID environment variables.
        See https://python.langchain.com/docs/integrations/tools/google_search.

        :param query: The search query to be used for the Google search.
        :return: The search results as a string.
        """
        from langchain.utilities import GoogleSearchAPIWrapper

        search = GoogleSearchAPIWrapper()
        return search.run(query)
    
class AgentV1(AgentV0, LangChainTools):
    pass
```

## Orchestration

One of ActionWeaver core feature is orchestration of actions. Each action has these two following attributes: 

**Scope**: Each action is confined to its own visibility scope.

**Orchestration expression**:

1. **SelectOne(['a1', 'a2', 'a3])**: Prompting the llm to choose either 'a2' or 'a3' after 'a1' has been invoked, or to take no action.
   
2. **RequireNext(['a1', 'a2', 'a3])**: Mandating the language model to execute 'a2' immediately following 'a1', followed by 'a3'.


### Example: Hierarchy of Actions

Instead of overwhelming OpenAI with an extensive list of functions, we can design a hierarchy of actions. In this example, we introduce a new class that defines three specific actions, reflecting a hierarchical approach:

- FileHandler with `default` scope: This action serves as the entry point for all file-manipulating actions, with orchestration logic `SelectOne(["FileHandler", "ListFiles", "ReadFile"])`.

- ListFiles with `file` scope.
- ReadFile with `file` scope.

```python
class FileUtility(AgentV0):
    @action(name="FileHandler", orch_expr = SelectOne(["FileHandler", "ListFiles", "ReadFile"]))
    def handle_file(self, instruction: str) -> str:
        """
        Handles user instructions related to file operations. Put every context in the instruction only!
    
        Args:
            instruction (str): The user's instruction about file handling.
    
        Returns:
            str: The response to the user's question.
        """
        return instruction
        

    @action(name="ListFiles", scope="file")
    def list_all_files_in_repo(self, repo_path: str ='.') -> List:
        """
        Lists all the files in the given repository.
    
        :param repo_path: Path to the repository. Defaults to the current directory.
        :return: List of file paths.
        """

        logger.info(f"list_all_files_in_repo: {repo_path}")
        
        file_list = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_list.append(os.path.join(root, file))
            break
        return file_list

    @action(name="ReadFile", scope="file")
    def read_from_file(self, file_path: str) -> str:
        """
        Reads the content of a file and returns it as a string.
    
        :param file_path: The path to the file that needs to be read.
        :return: A string containing the content of the file.
        """
        logger.info(f"read_from_file: {file_path}")
        
        with open(file_path, 'r') as file:
            content = file.read()
        return f"The file content: \n{content}"
```

### Example: Chains of Actions

We can also force LLM to ask for current time after read a file by setting orchestration in `ReadFile`.

```python
class FileUtility(AgentV0):
    @action(name="FileHandler", orch_expr = SelectOne(["FileHandler", "ListFiles", "ReadFile"]))
    def handle_file(self, instruction: str) -> str:
        """
        Handles user instructions related to file operations. Put every context in the instruction only!
    
        Args:
            instruction (str): The user's instruction about file handling.
    
        Returns:
            str: The response to the user's question.
        """
        return instruction
        

    @action(name="ListFiles", scope="file")
    def list_all_files_in_repo(self, repo_path: str ='.') -> List:
        """
        Lists all the files in the given repository.
    
        :param repo_path: Path to the repository. Defaults to the current directory.
        :return: List of file paths.
        """

        logger.info(f"list_all_files_in_repo: {repo_path}")
        
        file_list = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_list.append(os.path.join(root, file))
            break
        return file_list

    @action(name="ReadFile", scope="file", orch_expr = RequireNext(["ReadFile", "GetCurrentTime"]))
    def read_from_file(self, file_path: str) -> str:
        """
        Reads the content of a file and returns it as a string.
    
        :param file_path: The path to the file that needs to be read.
        :return: A string containing the content of the file.
        """
        logger.info(f"read_from_file: {file_path}")
        
        with open(file_path, 'r') as file:
            content = file.read()
        return f"The file content: \n{content}"
```

## Rewinding Actions
WIP