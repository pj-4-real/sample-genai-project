import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.1
)

parser = StrOutputParser()

def chat():
    chat_history = [
        ("system", "You are a helpful chatbot. Give concise and accurate responses.")
    ]
    print("Langchain Chatbot. type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        chat_history.append(("user", user_input))

        # Building the prompt dynamically
        prompt = ChatPromptTemplate.from_messages(chat_history)
        chain = prompt | llm | parser

        # Get response for the user query
        response = chain.invoke({})

        # Save and show the response
        print(f"Bot: {response}\n")
        chat_history.append(("assistant", response))

        print("-" * 80)


chat()