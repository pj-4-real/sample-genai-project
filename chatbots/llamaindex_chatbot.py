import os
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.groq import Groq
from dotenv import load_dotenv

load_dotenv()
llm = Groq(
    groq_api_key=os.envion["GROQ_API_KEY"],
    model="openai/gpt-oss-120b",
    temperature=0
)

def chat():
    history = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful chatbot. Give concise and accurate responses.")
    ]
    print("Llamaindex Chatbot. type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        history.append(ChatMessage(role=MessageRole.USER, content=user_input))

        # Building the prompt dynamically
        resp = llm.chat(messages=history)
        answer = resp.message.content

        # Save and show the response
        print(f"Bot: {answer}\n")
        history.append(ChatMessage(role=MessageRole.ASSISTANT, content=answer))

        print("-" * 80)

