# LLM Configuration for Graph Reasoning Experiments

# Model definitions
MODELS = {
    # Local Ollama models (recommended - no quota limits)
    "llama3.2-1b": {
        "provider": "ollama",
        "ollama_name": "llama3.2:1b",
        "size": "1.3GB",
        "params": "1.2B",
        "temperature": 0.0,
        "status": "completed"  # Already ran 600/600 Facebook prompts
    },
    "qwen2.5-3b": {
        "provider": "ollama",
        "ollama_name": "qwen2.5:3b",
        "size": "1.9GB",
        "params": "3B",
        "temperature": 0.0,
        "status": "completed"  # Already ran 600/600 Facebook prompts
    },
    "phi3-medium": {
        "provider": "ollama",
        "ollama_name": "phi3:medium",
        "size": "7.9GB",
        "params": "14B",
        "temperature": 0.0,
        "ram_required": "30.7GB",
        "status": "partial",  # Ran 150/600 (OOM on system with 23.5GB RAM)
        "note": "Requires 30GB+ RAM, skip if insufficient"
    },
    
    # OpenAI models (disabled - quota exhausted)
    # "gpt-4o-mini": {
    #     "provider": "openai",
    #     "temperature": 0.0,
    #     "status": "quota_exceeded",  # 163/600 completed, 80% null
    # },
    # "gpt-3.5-turbo": {
    #     "provider": "openai", 
    #     "temperature": 0.0,
    #     "status": "quota_exceeded",  # 600/600 all null (Error 429)
    # },
}

# Active models for NLGraph experiments
ACTIVE_MODELS = ["llama3.2-1b", "qwen2.5-3b", "phi3-medium"]

# OpenAI API key (if needed)
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
