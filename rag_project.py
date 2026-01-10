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

question = "Worum geht es in diesen Dokumenten?"

retrieved_chunks=retriever.retrieve(question)

context = "\n".join(chunk["text"] for chunk in retrieved_chunks)


answer = generate_answer(question, retrieved_chunks)

# Ergebnisanzeige

print("Frage:")
print(question)
print("\nGefundener Kontext:")
print(context)
print("\nAntwort:")
print(answer)

print("\n📚 Quellen:")
for c in retrieved_chunks:
    print(f"- {c['source']} | Seite: {c['page']} | Chunk: {c['chunk_id']}")
 
