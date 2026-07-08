import os
import time
import json
import random
from openai import OpenAI
from dotenv import load_dotenv

# Load .env from project root (one level up from scripts/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ──────────────────────────────────────────────
# Load OpenAI key
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)

# ──────────────────────────────────────────────
# OpenAI LLM Call
# ──────────────────────────────────────────────
def call_openai_model(model_name, prompt, max_tokens=800, temperature=0):
    """
    Generic wrapper for OpenAI chat models.
    Includes retry + timeout protection.
    """

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            return output

        except Exception as e:
            print(f"[OpenAI Retry {attempt+1}/5] Error:", e)
            time.sleep(2 + attempt)

    return None

# ──────────────────────────────────────────────
# OLLAMA Local Models
# ──────────────────────────────────────────────
def call_ollama_model(model_name, prompt, max_retries=3):
    """
    Call Ollama models via REST API (port 11434).
    Requires Ollama server to be running: ollama serve
    """
    import requests
    
    # Increase timeout for larger models like phi3-medium
    timeout = 300  # 5 minutes for large models
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"[Ollama] HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                print(f"[Ollama] Connection error. Make sure Ollama is running: 'ollama serve'")
            time.sleep(2)
        except Exception as e:
            print(f"[Ollama Retry {attempt+1}/{max_retries}] Error:", e)
            time.sleep(2)
    
    return None

# ──────────────────────────────────────────────
# Save record
# ──────────────────────────────────────────────
def save_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
