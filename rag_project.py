from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Antworte mit genau einem Satz: Was ist RAG?"
)

print(response.output_text)