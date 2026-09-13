import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
response = client.responses.create(model="gpt-3.5-turbo", input="Give me a three sentence story about a unicorn")

print(response)
print(response.output[0].content[0].text)