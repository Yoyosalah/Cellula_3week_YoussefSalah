from langchain_community.document_loaders import TextLoader
from pathlib import Path

def load_document(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"[Loader] Loaded {len(documents)} document(s) from '{file_path}'")
    return documents

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    file_path = BASE_DIR / "data" / "youssef_salah_profile.txt"

    print(load_document(str(file_path)))