import os
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
# from llama_index.llms.groq import Groq
# Uncomment to use groq

#LLM Setup
def run_llm(context: str, query: str) -> str:
    llm = Ollama(model="gemma2:2b", temperature=0.0)

    template_str = (
        "You are an expert AI assistant.\n"
        "Use ONLY the provided context to answer the user's queries."
        "If the context is insufficient or doesn't mention the answer, reply exactly: "
        "'Not enough information.'\n\n"
        "Context:\n{context_str}\n\n"
        "User Question: {query_str}\n\n"
        "Answering Rules:\n"
        "1) Be concise and precise (3-6 sentences, unless the question requires more).\n"
        "2) Use bullet points for lists.\n"
        "3) At the end, include a 'Sources:' section with short snippets or filenames from the context you used.\n\n"
        "Final Answer:"
    )

    template = PromptTemplate(template_str)
    filled_prompt = template.format(context_str=context, query_str=query)

    response = llm.complete(prompt=filled_prompt)
    return response.text

context = (
    "Transformers use a self-attention mechanism that lets each token attend "
    "to every other toke in the sequence. This enables modeling of long-range "
    "dependencies without recurrence. Positional encodings inject order "
    "information, and multi-head attention captures diverse relations.\n\n"
    "the encoder stacks layers of self-attention and feed-forward networks to "
    "build contextual representations. The decoder uses masked self-attention "
    "to maintain causality and cross-attention to consult encoder outputs"
)
query = "How do transformers handler long-range dependencies?"

print(run_llm(context, query))


