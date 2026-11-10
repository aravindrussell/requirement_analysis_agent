import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-6.1-sol")
CLAUDE_MODEL_DEFAULT = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def has_anthropic_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))
