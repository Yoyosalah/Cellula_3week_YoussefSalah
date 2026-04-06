from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from pathlib import Path

# Load app.env from the project root (one level up from src/)
load_dotenv(Path(__file__).resolve().parent.parent / "app.env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions about Youssef Salah.
Use ONLY the context provided below to answer. If the answer is not found in the context,
say: "I don't have that information in my profile."

Context:
{context}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, model_name: str = "arcee-ai/trinity-large-preview:free"):
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"[RAG Chain] Built RAG chain with OpenRouter model '{model_name}'")
    return rag_chain