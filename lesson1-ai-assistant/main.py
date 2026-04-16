from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

while True:
    user_input = input("You: ")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a car mechanic assistant"},
            {"role": "user", "content": user_input}
        ]
    )

    print("AI:", response.choices[0].message.content)