from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np

load_dotenv()
client = OpenAI()

documents = [
    "RAG steht für Retrieval Augmented Generation.",
    "RAG kombiniert Vektorsuche mit großen Sprachmodellen.",
    "RAG reduziert Halluzinationen, indem externe Quellen genutzt werden.",
    "FAISS ist eine Bibliothek für schnelle Ähnlichkeitssuche von Vektoren.",
    "Embeddings repräsentieren Text als Zahlenvektoren."
]


embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents
)

doc_vectors=np.array(
    [item.embedding for item in embedding_response.data],
    dtype='float32'
)

# Faiss Index erstellen

dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension) 
index.add(doc_vectors)

context = "\n".join(documents[:2])

# Nutzerfrage

question = "Warum ist RAG nützlich?"

# Berechne Embeddings für die Nutzerfrage

question_embedding= client.embeddings.create(
    model="text-embedding-3-small",
    input=[question]
)

query_vector = np.array(
    [question_embedding.data[0].embedding],
    dtype='float32'
)

# Ähnlichkeitssuche (Top-k)

k=2
distances, indices = index.search(query_vector, k)

retrieved_docs = [documents[i] for i in indices[0]]

context="\n".join(retrieved_docs)

# Prompt mit gefundenem Kontext

prompt = f"""
Beantworte die Frage nur anhand des Kontexts.

Kontext: 
{context}

Frage:
{question}
"""
# Anfrage an das LLM
response = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt,
    max_output_tokens=150
)

# Ergebnisanzeige

print("Frage:")
print(question)
print("\nGefundener Kontext:")
print(context)
print("\nAntwort:")
print(response.output_text)

