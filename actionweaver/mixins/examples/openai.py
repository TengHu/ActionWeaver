# from openai import OpenAI

# client = OpenAI()

# from actionweaver import action


# class OpenAIAPI:
#     @action(name="GenerateImage")
#     def generate_image(self, prompt: str) -> str:
#         """
#         Generates an image based on a given prompt.

#         Args:
#             prompt (str): A descriptive text input that guides the content of the generated image (e.g., "lion running on Mars").

#         Returns:
#             str: A message indicating that the image has been created, along with the URL of the generated image.
#         """
#         from IPython.display import Image, display

#         response = client.images.generate(prompt=prompt, n=1, size="1024x1024")

#         url = response.data[0].url
#         display(Image(url=url))
#         return "The image has been created"
