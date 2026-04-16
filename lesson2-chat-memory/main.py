from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

messages = [
    {"role": "system", "content": "You are a helpful assistant"}
]

while True:
    user_input = input("You: ")
    
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    ai_message = response.choices[0].message.content
    
    print("AI:", ai_message)
    
    
    messages.append({"role": "assistant", "content": ai_message})
