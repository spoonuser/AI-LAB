import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
chunks = text.split("\n\n")
valid_chunks = [chunk for chunk in chunks if chunk.strip()]

def get_embedding(text):
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def preprocess(text):
    return text.lower()

chunk_embeddings = [
    get_embedding(preprocess(chunk))
    for chunk in valid_chunks
]

def cosine_similarity(a, b):
    return np.dot(a ,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_chunks(query, top_k=3):
    query_embedding = get_embedding(preprocess(query))
    
    scores = []
    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, valid_chunks[i]))
        
    scores.sort(reverse=True)
    return [chunk for _, chunk in scores[:top_k]]

rag_intent = """
how much does it cost
price of service
oil change price
car repair price
bmw repair
toyota service
diagnostics cost
location of service
working hours
"""

rag_intent_embedding = get_embedding(preprocess(rag_intent))

def is_rag_query(query, threshold=0.4):
    query_embedding = get_embedding(preprocess(query))
    
    score = cosine_similarity(query_embedding, rag_intent_embedding)
    
    print("DEBUG SCORE:", score)
    
    return score > threshold

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant"
    }
]

while True:
    user_input = input("You: ")
    use_rag = is_rag_query(user_input)
    
    if use_rag:
        relevant_chunks = find_relevant_chunks(user_input)
        context_text = "\n".join(relevant_chunks)
        
        print("DEBUG CONTEXT:", context_text)
        
        prompt = f"""
You MUST answer using ONLY the context below.

Context:
{context_text}

Question:
{user_input}

if answer is not in context, say "I don't know."
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
