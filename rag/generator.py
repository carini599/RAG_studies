from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()

def generate_answer(question, context):
    
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

    return response.output_text