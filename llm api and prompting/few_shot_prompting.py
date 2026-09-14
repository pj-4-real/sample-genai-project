# - FEW-SHOT -
from typing import List, Dict
import os
from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq
from dotenv import load_dotenv

# 1. Load variables from .env
load_dotenv()

def run_llm_fewshot(context: str, query: str, examples: List[Dict[str, str]]) -> str:
    llm = Groq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    # 2. Format the examples into a single string
    examples_str = "\n".join(
        f"Example {i + 1}:\n"
        f"Context: {ex.get('context', '')}\n"
        f"Question: {ex.get('question', '')}\n"
        f"Answer: {ex.get('answer', '')}\n"
        for i, ex in enumerate(examples)
    )

    template_str = (
        "You are an expert AI assistant.\n"
        "Use ONLY the provided context to answer the user's question.\n"
        "If the context is insufficient or does not mention the answer, reply exactly: "
        "'Not enough information.'\n\n"
        "Follow the style and reasoning illustrated by the examples.\n\n"
        "Examples:\n{examples_str}\n"
        "--- End of Examples ---\n\n"
        "Context:\n{context_str}\n\n"
        "User Question:\n{query_str}\n\n"
        "Answering Rules:\n"
        "1) Be concise and precise (3-6 sentences, unless the question requires more).\n"
        "2) Use bullet points for lists.\n"
        "3) At the end, include a 'Sources:' section with short snippets or filenames from the context you used.\n\n"
        "Final Answer:"
    )

    # 3. Pass 'examples_str' to the formatting engine, not the raw 'examples' list
    prompt = PromptTemplate(template_str).format(
        examples_str=examples_str,
        context_str=context,
        query_str=query
    )

    response = llm.complete(prompt=prompt)
    output = response.text
    return output


# --- Example Usage ---

# Define the historical examples to teach the AI how to format its answers
training_examples = [
    {
        "context": "XYZ provides 15 days of paid time off (PTO) per calendar year. Unused PTO expires on December 31st and does not roll over.",
        "question": "How many PTO days do I get and do they carry over to next year?",
        "answer": "You receive 15 days of paid time off (PTO) annually. Any unused PTO expires at the end of the calendar year and does not roll over.\n\nSources: '15 days of paid time off', 'does not roll over.'"
    },
    {
        "context": "Employees must submit an IT Portal ticket for hardware replacements. Laptops are only eligible for a refresh if they are at least 3 years old.",
        "question": "How do I get a new laptop?",
        "answer": "To request a new laptop, you must submit a ticket through the IT Portal. Replacements are only approved if your current machine is at least 3 years old.\n\nSources: 'submit an IT Portal ticket', 'at least 3 years old.'"
    }
]

# The actual question you want to ask right now
sample_context = "XYZ requires associates to be billed to a project for a minimum of 225 business days out of a 12-month period. Unallocated time beyond 35 days will flag an HR review."
sample_query = "What is the XYZ deployment policy?"

# Run the function
print(run_llm_fewshot(context=sample_context, query=sample_query, examples=training_examples))