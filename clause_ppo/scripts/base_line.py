from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv()

client = InferenceClient(
    token=os.getenv("qwen_token")
)

messages = [{
    "role": "user",
    "content": "Please generate a SQL query according to the given question and schema.",
}]

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct:featherless-ai",
    messages=messages,
    max_tokens=500,
)

print(completion.choices[0].message.content)
