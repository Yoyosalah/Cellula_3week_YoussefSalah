from langchain_text_splitters import NLTKTextSplitter

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = NLTKTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    from loader import load_document
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    file_path = BASE_DIR / "data" / "youssef_salah_profile.txt"

    docs = load_document(str(file_path))
    chunks = split_documents(docs)
    print(chunks[0].page_content[:500])
    print(len(chunks))