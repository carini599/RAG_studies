from dotenv import load_dotenv

load_dotenv()

import faiss
import numpy as np
from openai import OpenAI

client=OpenAI()

class Retriever:
    def __init__(self, chunks):
        
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        self.vectors =np.array(
            [item.embedding for item in embedding_response.data],
            dtype = "float32"
        )

        dimension = self.vectors.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(self.vectors)
    
    def retrieve(self, query, k=5, max_distance=1.2):
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )

        query_vector = np.array(
            [query_embedding.data[0].embedding],
            dtype="float32"
        )

        distances, indices = self.index.search(query_vector, k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if distance <= max_distance:
                chunk = self.chunks[idx]
                chunk["distance"] = float(distance)
                results.append(chunk)

        return results