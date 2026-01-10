from pathlib import Path
from pypdf import PdfReader
import tiktoken

DATA_DIR = Path("data")
CHUNK_SIZE = 400 
CHUNK_OVERLAP = 50


def load_documents():
    
    chunks = []
    chunk_id = 0

    for file in DATA_DIR.iterdir():
        if file.suffix == ".pdf":
            chunks.extend(load_pdf(file, chunk_id))
            chunk_id = len(chunks)
        elif file.suffix in [".txt", ".md"]:
            chunks.extend(load_text(file, chunk_id))

    return chunks


def load_pdf(path: Path, start_chunk_id:int):
    reader = PdfReader(path)
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    chunk_id = start_chunk_id

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        tokens = enc.encode(text)
        start = 0

        while start < len(tokens):
            end = start + CHUNK_SIZE
            chunk_tokens = tokens[start:end]

            chunks.append({
                "text": enc.decode(chunk_tokens),
                "source": path.name,
                "page": page_number,
                "chunk_id": chunk_id
            })

            chunk_id += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def load_text(path: Path, start_chunk_id: int):
    enc = tiktoken.get_encoding("cl100k_base")
    text = path.read_text(encoding="utf-8")
    tokens = enc.encode(text)

    chunks = []
    chunk_id = start_chunk_id
    start = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]

        chunks.append({
            "text": enc.decode(chunk_tokens),
            "source": path.name,
            "page": None,
            "chunk_id": chunk_id
        })

        chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

