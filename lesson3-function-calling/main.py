import os
import datetime
from openai import OpenAI
import json

# API клиент
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# История диалога
messages = [
    {"role": "system", "content": "You are a helpful assistant. ALWAYS use the get_current_time function when user asks about time. ALWAYS use the add_numbers function when user asks about the sum of 2 numbers. ALWAYS use the multiply_numbers function when user asks about the multiplication of 2 numbers"}
]
# Функция
def get_current_time():
    return str(datetime.datetime.now())

def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b

# Описание инструмента
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current date and time",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Get the sum of 2 numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a":{
                        "type":"number",
                        "description":"First number"
                    },
                    "b":{
                        "type":"number",
                        "description":"Second number"
                    }
                },
                "required":["a","b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply_numbers",
            "description": "Get the multiplication of 2 numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a":{
                        "type":"number",
                        "description":"First number"
                    },
                    "b":{
                        "type":"number",
                        "description":"Second number"
                    }
                },
                "required":["a","b"]
            }
        }
    }
]

while True:
    user_input = input("You: ")

    messages.append({
        "role": "user",
        "content": user_input
    })

    # Первый вызов LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )

    msg = response.choices[0].message

    # 🔥 ВАЖНО: всегда добавляем ответ ассистента
    messages.append(msg)

    # Если есть вызов функции
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "get_current_time":
                result = get_current_time()

                # Добавляем результат функции
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            elif tool_call.function.name == "add_numbers":
                args = json.loads(tool_call.function.arguments)
                
                a = args["a"]
                b = args["b"]
                
                result = add_numbers(a , b)

                # Добавляем результат функции
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
            elif tool_call.function.name == "multiply_numbers":
                args = json.loads(tool_call.function.arguments)
                
                a = args["a"]
                b = args["b"]
                
                result = multiply_numbers(a , b)

                # Добавляем результат функции
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        # Второй вызов LLM (с результатом функции)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        final_msg = response.choices[0].message
        print("AI:", final_msg.content)

        messages.append({
            "role": "assistant",
            "content": final_msg.content
        })

    else:
        # Обычный ответ
        print("AI:", msg.content)