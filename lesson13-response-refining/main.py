import os
from openai import OpenAI
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

user_sessions = {}

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

def clean_answer(text):
    return text.strip().replace("\n\n", "\n")

chunk_embeddings = [
    get_embedding(preprocess(chunk))
    for chunk in valid_chunks
]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_chunks(query, top_k = 3):
    query_embedding = get_embedding(preprocess(query))
    
    scores = []
    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, valid_chunks[i]))
        
    scores.sort(reverse=True)
    return [chunk for _, chunk in scores[:top_k]]

rag_intent = """
oil change price
service cost
car repair price
bmw repair
toyota service
diagnostics cost
location almaty working hours
"""

rag_intent_embedding = get_embedding(preprocess(rag_intent))

SYSTEM_PROMPT = """
You are a professional AI assistant for a car service business.

Your goals:
- Be clear, helpful, and confident
- Answer like a real human employee
- Keep answers short but informative
- If user asks about price → give exact price
- If you don't know → say "I don't know"

Style:
- Friendly but professional
- No sarcasm
- No jokes
- No unnecessary text

Always:
- Answer directly
- Don't explain how you think
"""

def get_user_messages(user_id):
    if user_id not in user_sessions:
        user_sessions[user_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    return user_sessions[user_id]

def is_raq_query(query, threshold=0.4):
    query_embedding = get_embedding(preprocess(query))
    
    score = cosine_similarity(query_embedding, rag_intent_embedding)
    
    print("DEBUG SCORE:", score)
    
    return score > threshold

async def handle_message(update: Update, context:ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text
    
    messages = get_user_messages(user_id)
    
    use_rag = is_raq_query(user_input)
    
    if use_rag:
        relevant_chunks = find_relevant_chunks(user_input)
        context_text = "\n".join(relevant_chunks)
        
        prompt = f"""
You are answering based on company knowledge.

Context:
{context_text}

User question:
{user_input}

Rules:
- Use context if relevant
- If partial info → combine with general knowledge
- If not found → say "I don't know"
- Answer naturally like a human

Answer:
"""
    else:
        prompt = user_input
        
    messages.append({
        "role":"user",
        "content": prompt
    })
    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=150
)
    
    answer = response.choices[0].message.content
    
    answer = clean_answer(answer)
    
    messages.append({
        "role":"assistant",
        "content": answer
    })
    
    if len(messages) > 12:
        messages[:] =[messages[0]]+ messages[-10:]
        
    await update.message.reply_text(answer)
    
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print("Bot is running...")
app.run_polling()
    
