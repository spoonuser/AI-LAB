import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("knowledge.txt", "r", encoding="utf-8") as f:
    knowledge = f.read()
    
messages = [
    {
        "role":"system",
        "content":"You are a helpful assistant. Answer ONLY using the provided knowledge. If answer is not in knowledge, say you don't know"
    }
]

while True:
    user_input = input("You: ")
    
    context = f"""
Knowledge:
{knowledge}

Question:
{user_input}
"""
    messages.append({
        "role":"user",
        "content":context
    })
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    answer = response.choices[0].message.content
    print("AI:", answer)
    
    messages.append({
        "role":"assistant",
        "content":"answer"
    })