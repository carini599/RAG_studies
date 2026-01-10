from pathlib import Path
from pypdf import PdfReader
import tiktoken

DATA_DIR = Path("data")
CHUNK_SIZE = 400 
CHUNK_OVERLAP = 50


def load_documents():
    
    texts=[]
    
    for file in DATA_DIR.iterdir():
        if file.suffix == ".pdf":
            texts.extend(load_pdf(file))
        elif file.suffix in [".txt", ".md"]:
            text.extend(load_text(file))

    return chunk_texts(texts)


def load_pdf(path: Path):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return pages

def load_text(path: Path):
    return [path.read_text(encoding="utf-8")]


def chunk_texts(texts): 
    
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []

    for text in texts:
        # Change text to tokens
        tokens = enc.encode(text)

        start = 0
        while start < len(tokens):
            end = start + CHUNK_SIZE
            chunk_tokens = tokens[start:end]
            chunk_text=enc.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks