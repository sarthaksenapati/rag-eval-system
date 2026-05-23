from pathlib import Path
from llama_index.core import Document
import httpx
from pypdf import PdfReader

def load_pdf(file_path: str) -> list[Document]:
    """Extracts text from PDF page by page using pypdf."""
    reader = PdfReader(file_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():  # skip blank pages
            docs.append(Document(
                text=text.strip(),
                metadata={
                    "file_name": Path(file_path).name,
                    "page": i + 1,
                    "source": file_path,
                    "type": "pdf"
                }
            ))
    print(f"Loaded {len(docs)} pages from {Path(file_path).name}")
    return docs


def load_from_directory(dir_path: str) -> list[Document]:
    """Loads all supported files from a directory."""
    docs = []
    path = Path(dir_path)

    for file in path.rglob("*"):
        if file.suffix.lower() == ".pdf":
            docs.extend(load_pdf(str(file)))
        elif file.suffix.lower() in [".md", ".txt", ".html"]:
            text = file.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append(Document(
                    text=text.strip(),
                    metadata={
                        "file_name": file.name,
                        "source": str(file),
                        "type": file.suffix.lower().strip(".")
                    }
                ))
            print(f"Loaded: {file.name}")

    print(f"\nTotal document pages/chunks loaded: {len(docs)}")
    return docs


def load_from_urls(urls: list[str]) -> list[Document]:
    """Fetches raw text from a list of URLs."""
    docs = []
    for url in urls:
        try:
            response = httpx.get(url, timeout=None)
            response.raise_for_status()
            docs.append(Document(
                text=response.text,
                metadata={"source": url, "type": "web"}
            ))
            print(f"Loaded: {url}")
        except Exception as e:
            print(f"Failed to load {url}: {e}")
    return docs


def load_from_texts(texts: list[dict]) -> list[Document]:
    """Load from raw text + metadata dicts. Good for testing."""
    return [
        Document(
            text=item["text"],
            metadata={k: v for k, v in item.items() if k != "text"}
        )
        for item in texts
    ]