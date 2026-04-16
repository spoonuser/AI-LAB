import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("knowledge.txt", "r", encoding="utf=8") as f:
    text = f.read()

chunks = text.split("\n\n")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

chunk_embeddings =  [get_embedding(chunk) for chunk in chunks if chunk.strip()]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_chunks(query, top_k=5):
    query_embedding = get_embedding(query)
    
    scores = []
    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, chunks[i]))
    
    scores.sort(reverse=True)
    return [chunk for _, chunk in scores[:top_k]]

messages = [
    {
        "role":"system",
        "content": """
You are a helpful AI assistant.

If user asks general questions (like greetings, casual talk), answer normally.

If user asks about company, services, prices, or anything related to business, use the provided context.

If the answer is not in the context, say you don't know.
"""
    }
]

while True:
    user_input = input("You: ")
    
    relevant_chunks = find_relevant_chunks(user_input)
    
    context = "/n".join(relevant_chunks)
    
    prompt = f"""
Context:
{context}

Question:
{user_input}
"""

    messages.append({
        "role":"user",
        "content": prompt
    })
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    answer = response.choices[0].message.content
    print("AI:", answer)
    
    messages.append({
        "role": "assistant",
        "content": answer
    })
