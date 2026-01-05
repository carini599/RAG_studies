from dotenv import load_dotenv
from rag.loader import load_documents
from rag.retriever import Retriever
from rag.generator import generate_answer

# Import Env Variables
load_dotenv()

documents = load_documents()

retriever = Retriever(documents)

# Nutzerfrage

question = "Warum ist RAG nützlich?"

retrieved_docs=retriever.retrieve(question)

context="\n".join(retrieved_docs)

answer = generate_answer(question, context)

# Ergebnisanzeige

print("Frage:")
print(question)
print("\nGefundener Kontext:")
print(context)
print("\nAntwort:")
print(answer)

