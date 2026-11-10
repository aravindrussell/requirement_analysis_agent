from langchain.chat_models import init_chat_model

def build_chat_model(provider: str, model_name: str):
    """Create a LangChain chat model for the selected provider."""
    provider = provider.lower().strip()

    if provider == "openai":
        return init_chat_model(
            model=f"openai:{model_name}",
        )

    if provider == "claude":
        return init_chat_model(
            model=f"anthropic:{model_name}",
        )

    raise ValueError(f"Unsupported provider: {provider}")
