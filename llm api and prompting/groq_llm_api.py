import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

chat_completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is Deep Learning?"}
    ]
)

print(chat_completion.choices[0].message.content)