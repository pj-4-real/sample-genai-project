import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()
# response = client.models.generate_content(model="gemini-3.5-flash", contents="Explain how AI works")
# print(response.text)


chat = client.chats.create(model="gemini-3.5-flash")
resp = chat.send_message("I have two dogs in my house")
print(resp.text)

print("-"*50)

resp = chat.send_message("How many paws are in my house?")
print(resp.text)