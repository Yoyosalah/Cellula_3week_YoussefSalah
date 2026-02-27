import gradio as gr

from src.loader import load_document
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vectorstore import get_or_build_vectorstore,get_retriever
from src.rag_chain import build_rag_chain

from pathlib import Path

# Make all paths absolute relative to app.py's location
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "youssef_salah_profile.txt"

## Initializing RAG PIPELINE
print("Initializing RAG pipeline...")
documents = load_document(DATA_PATH)
chunks = split_documents(documents)
embeddings = get_embedding_model()
vectorstore = get_or_build_vectorstore(chunks, embeddings)
retriever = get_retriever(vectorstore, k=4)
rag_chain = build_rag_chain(retriever)
print("RAG pipeline ready.\n")


def answer_question(question: str, history: list):
    if not question.strip():
        return "Please enter a question."
    response = rag_chain.invoke(question)
    return response

# Gradio UI
with gr.Blocks(title="Ask About Youssef Salah", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 Ask Me Anything — Youssef Salah
        This chatbot answers questions about Youssef using a RAG system
        built with **LangChain + ChromaDB + HuggingFace Embeddings**.
        """
    )

    chatbot = gr.ChatInterface(
        fn=answer_question,
        chatbot=gr.Chatbot(height=450),
        textbox=gr.Textbox(
            placeholder="e.g. What are Youssef's skills? What projects has he built?",
            container=False,
            scale=7,
        ),
        examples=[
            "What is Youssef's educational background?",
            "What programming languages does he know?",
            "Tell me about his work experience.",
            "What are his career goals?",
            "What projects has he built?",
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
