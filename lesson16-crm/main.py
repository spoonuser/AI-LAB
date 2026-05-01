import os
from openai import OpenAI
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

user_sessions = {}

with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds_dict = {
  
}

creds = ServiceAccountCredentials.from_json_keyfile_dict(
    creds_dict, scope
)

client_gs = gspread.authorize(creds)

sheet = client_gs.open("AI Leads").sheet1

def save_lead(user_id, phone, message):
    from datetime import datetime
    
    sheet.append_row([
        str(user_id),
        phone,
        message,
        str(datetime.now())
    ])

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
You are an AI assistant for a car service.

Your goals:
- Help the user
- Answer clearly and professionally
- Gently guide user to book a service

Rules:
- If user asks about service → suggest booking
- If user shows interest → ask for phone number
- Do NOT be pushy
- Be natural like a human manager

Style:
- Short
- Friendly
- Professional
"""
user_states = {}

def set_state(user_id, state):
    user_states[user_id] = state
    
def get_state(user_id):
    return user_states.get(user_id, "NEW")
    

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

def is_ready_to_convert(text):
    keywords = ["book", "appointment", "come", "when", "time", "schedule"]
    return any(k in text.lower() for k in keywords)
  
def extract_phone(text):
    match = re.search(r'\+?\d{10,15}', text)
    return match.group(0) if match else None

async def handle_message(update: Update, context:ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_state(user_id)
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
    if is_ready_to_convert(user_input):
        set_state(user_id, "ASK_PHONE")
        await update.message.reply_text("Very good. Now tell me ur phone number.")
        #prompt += "\nAsk user for phone number to confirm booking."
        return
    if state == "ASK_PHONE":
        phone = extract_phone(user_input)
        
        if phone:
            save_lead(user_id, phone, user_input)
            set_state(user_id, "ASK_TIME")
            await update.message.reply_text("What time is preferable for you?")
        else:
            await update.message.reply_text("Pls, send me the correct number.")
        
        return
    
    if state == "ASK_TIME":
        set_state(user_id, "DONE")
        
        await update.message.reply_text(
            f"Done, booked for {user_input}. Will be waiting!"
        )
        
        return
    
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
    
