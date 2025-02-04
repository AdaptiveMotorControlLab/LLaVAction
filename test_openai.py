from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = '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'
# client = OpenAI()

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Hello."
#         }
#     ]
# )

# print(completion.choices[0].message)


API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
GPT_EVAL_MODEL_NAME = 'gpt-3.5-turbo-0613'
max_tokens = 64

messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Hello."
        }
    ]

headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        # "response_format": {"type": "json_object"},
    }

response = requests.post(API_URL, headers=headers, json=payload, timeout=60)