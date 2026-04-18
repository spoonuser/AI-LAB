import os
from openai import OpenAI
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = "8733685125:AAGkfovG7U8xfdG3lMFXWtDu_Atrou3msWI"

client = OpenAI(api_key=OPENAI_KEY)

with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
chunks = text.split("\n\n")
valid_chunks = [chunk for chunk in chunks if chunk.strip()]

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
    for chunk in valid_chunks
]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_chunks(query, top_k=5):
    query_embedding = get_embedding(preprocess(query))
    
    scores = []
    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, valid_chunks[i]))
        
    scores.sort(reverse=True)
    return [chunk for _, chunk in scores[:top_k]]\

def classify_query(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a strict classifier.

If the question is about:
- services
- prices
- repair
- company
- cars

Return ONLY: RAG

If it's casual conversation (hello, how are you, jokes):
Return ONLY: GENERAL
"""
            },
            {
                "role":"user",
                "content": query
            }
        ]
    )
    return response.choices[0].message.content.strip()

messages = [
    {
        "role":"system", "content": "You are a helpful assistant."
    }
]

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    
    query_type = classify_query(user_input)
    
    if query_type == "RAG":
        relevant_chunks = find_relevant_chunks(user_input)
        context_text = "\n".join(relevant_chunks)
        
        prompt = f"""
Use this context:
    
{context_text}

Question:
{user_input}

If answer not found, say you don't know.
"""
    else:
        prompt = user_input
        
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    answer = response.choices[0].message.content
    
    messages.append({"role": "assistant", "content": answer})
    
    await update.message.reply_text(answer)

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
print("Bot is running...")
app.run_polling()

