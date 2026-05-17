from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="dummy",
    timeout=120
)

response = client.chat.completions.create(
    model="qwen3:14b",
    messages=[
        {"role":"user","content":"how to solve the problem of overfitting?"}
    ]
)

print(response.choices[0].message.content)