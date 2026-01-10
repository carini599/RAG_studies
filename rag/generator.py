from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()

def generate_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return "Keine relevanten Quellen gefunden."

    context_blocks = []
    for i, c in enumerate(retrieved_chunks, start=1):
        block = f"""
        [{i}] Quelle: {c['source']} | Seite: {c['page']}
        {c['text']}"""
        context_blocks.append(block)

    context = "\n".join(context_blocks)
    
    prompt = f"""
    Beantworte die Frage ausschließlich anhand der nummerierten Quellen.
    Verwende in deiner Antwort explizite Zitate wie [1], [2].
    Wenn die Information nicht enthalten ist, sage das klar.

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

    return response.output_text