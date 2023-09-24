# Installation
## Installation through pip

To install ActionWeaver run:

```bash
pip install actionweaver
```


## Environment setup

ActionWeaver requires an OpenAI account and API key. Once we have a key we'll want to set it as an environment variable by running:

```bash
export OPENAI_API_KEY="..."
```
Or pass the key in OpenAI directly via the api:

```python
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
```
