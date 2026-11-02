
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

try:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are FreightDoc AI, a logistics document assistant."
            },
            {
                "role": "user",
                "content": "Explain what a Bill of Lading is in 2 sentences."
            }
        ],
        temperature=0.3,
        max_completion_tokens=200
    )

    answer = response.choices[0].message.content

    print("\nFreightDoc AI - Groq Connection Successful!")
    print("------------------------------------------")
    print(answer)

except Exception as e:
    print("\nGroq API connection failed!")
    print("Error:", e)
