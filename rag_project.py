from dotenv import load_dotenv
from rag.loader import load_documents
from rag.retriever import Retriever
from rag.generator import generate_answer

# Import Env Variables
load_dotenv()

documents = load_documents()
print(f'Geladene Chunks: {len(documents)}')

retriever = Retriever(documents)

# Nutzerfrage

question = "Was kostet der Klarinettenunterricht in der 2er Gruppe pro Monat?"

retrieved_chunks = retriever.retrieve(
    question,
    k=5,
    max_distance=1.2
)

context = "\n".join(chunk["text"] for chunk in retrieved_chunks)


answer = generate_answer(question, retrieved_chunks)

# Ergebnisanzeige

print("Frage:")
print(question)
print("\n🔎 Gefundener Kontext nach Threshold:")
for i, c in enumerate(retrieved_chunks, start=1):
    print(f"[{i}] {c['source']} | Seite {c['page']} | Distanz {c['distance']:.3f}")
print("\nAntwort:")
print(answer)

print("\n📚 Quellen:")
for c in retrieved_chunks:
    print(f"- {c['source']} | Seite: {c['page']} | Chunk: {c['chunk_id']}")


 
