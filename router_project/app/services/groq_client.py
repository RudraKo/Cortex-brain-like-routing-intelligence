import os
import time
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

_raw_key = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = _raw_key.strip() if _raw_key else None

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")


@lru_cache(maxsize=1)
def _get_client():
    from groq import Groq

    return Groq(api_key=GROQ_API_KEY)

def generate_completion(prompt: str, model: str, system_message: str = None):
    """
    Sends a prompt to the specified Groq model and measures latency and tokens.
    Optionally accepts a system message to prime the model's behavior.
    """
    start_time = time.time()

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    client = _get_client()
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )

    end_time = time.time()
    latency_sec = end_time - start_time

    response_text = chat_completion.choices[0].message.content
    usage = chat_completion.usage

    return {
        "response": response_text,
        "model": model,
        "latency_sec": latency_sec,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }
