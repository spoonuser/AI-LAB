import os
import numpy as np
from openai import OpenAI

client =  OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
chunks = text.split("\n\n")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def preprocess(text):
    return text.lower()

chunk_embeddings = [
    get_embedding(preprocess(chunk))
    for chunk in chunks if chunk.strip()
]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_chunks(query, top_k = 3):
    query_embedding = get_embedding(preprocess(query))
    
    scores = []
    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, chunks[i]))
        
    scores.sort(reverse=True)
    return [chunk for _, chunk in scores[:top_k]]

def classify_query(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
Classify user query into:
- RAG (questions about company, services, prices, location)
- GENERAL (greetings, casual talk)

Answer ONLY with RAG or GENERAL.
"""
            },
            {
                "role": "system",
                "content": query
            }
        ]   
    )
    
    return response.choices[0].message.content.strip()

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]

while True:
    user_input = input("You: ")
    
    query_type = classify_query(user_input)
    print("DEBUG:", query_type)
    
    if query_type == "RAG":
        relevant_chunks = find_relevant_chunks(user_input)
        context = "\n".join(relevant_chunks)
        
        prompt = f"""
Use the context below to answer.

Context:
{context}

Question:
{user_input}
If answer is not in context, say you don't know.
"""
    else:
        prompt = user_input
        
    messages.append({
        "role": "user",
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