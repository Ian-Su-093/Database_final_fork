from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    token="your-token-here"
)

messages = [{
    "role": "user",
    "content": "Please generate a SQL query according to the given question and schema."
}]

completion = client.chat.completions.create(
    model="qwen/qwen2.5-coder-1.5b",
    messages=messages,
    max_tokens=500,
)

print(completion.choices[0].message.content)
