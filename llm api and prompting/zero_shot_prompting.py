# - ZERO-SHOT -
from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq
from dotenv import load_dotenv

load_dotenv()

def run_llm_zeroshot(context: str, query: str) -> str:
    llm = Groq(model="openai/gpt-oss-120b", temperature=0)

    template_str = (
        "You are an expert AI assistant.\n"
        "Use ONLY the provided context to answer the user's question.\n"
        "If the context is insufficient or does not mention the answer, reply exactly: "
        "'Not enough information.'\n\n"
        "Context:\n{context_str}\n\n"
        "User Question:\n{query_str}\n\n"
        "Answering Rules:\n"
        "1) Be concise and precise (3-6 sentences, unless the question requires more).\n"
        "2) Use bullet points for lists.\n"
        "3) At the end, include a 'Sources:' section with short snippets or filenames from the context you used.\n\n"
        "Final Answer:"
    )

    prompt = PromptTemplate(template_str).format(context_str=context, query_str=query)
    response = llm.complete(prompt=prompt)

    output = response.text
    return output

# --- Example Usage ---
sample_context = "XYZ requires associates to be billed to a project for a minimum of 225 business days out of a 12-month period."
sample_query = "What is the XYZ deployment policy?"
print(run_llm_zeroshot(sample_context, sample_query))