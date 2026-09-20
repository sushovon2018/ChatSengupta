"""Build the local Chroma RAG index from data/cricket Markdown files."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data/cricket")
DB_DIR = Path("data/chroma")


def main() -> None:
    files = sorted(DATA_DIR.rglob("*.md"))
    if not files:
        raise SystemExit("No Markdown files found in data/cricket")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    documents = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for document in splitter.create_documents([text], metadatas=[{"source": str(path)}]):
            documents.append(document)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="cricket_knowledge",
        persist_directory=str(DB_DIR),
    )
    print(f"Indexed {len(documents)} chunks from {len(files)} cricket files into {DB_DIR}")


if __name__ == "__main__":
    main()