"""Configuration for LLM and embedding providers.

Supports switching between OpenAI and LMStudio (local) providers via environment variables.

Environment Variables:
    LLM_PROVIDER: "openai" or "lmstudio" (default: "lmstudio")
    OPENAI_API_KEY: Required if using OpenAI
    LMSTUDIO_BASE_URL: LMStudio endpoint (default: "http://localhost:1234/v1")

    EMBEDDING_PROVIDER: "openai", "lmstudio", or "kusto" (default: "lmstudio")
    OPENAI_EMBEDDING_MODEL: OpenAI embedding model (default: "text-embedding-3-small")
    LMSTUDIO_EMBEDDING_MODEL: LMStudio embedding model name
"""

import os
from functools import lru_cache
from typing import Callable, Literal

from langchain_openai import ChatOpenAI

LLMProvider = Literal["openai", "lmstudio"]
EmbeddingProvider = Literal["openai", "lmstudio", "kusto"]


def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "lmstudio").lower()
    return provider  # type: ignore


def get_embedding_provider() -> EmbeddingProvider:
    """Get the configured embedding provider."""
    provider = os.getenv("EMBEDDING_PROVIDER", "lmstudio").lower()
    return provider  # type: ignore


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Create and return LLM instance based on configuration.

    Returns:
        ChatOpenAI configured for either OpenAI or LMStudio
    """
    provider = get_llm_provider()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,  # type: ignore
            temperature=0.7,
        )
    else:  # lmstudio (default)
        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        return ChatOpenAI(
            base_url=base_url,
            api_key="lm-studio",  # type: ignore - LMStudio doesn't need a real key
            temperature=0.7,
        )


def get_embedding_function() -> Callable[[str], tuple[list[float], str]]:
    """Create and return embedding function based on configuration.

    Returns:
        Callable that takes text and returns (embedding_vector, model_name)
    """
    provider = get_embedding_provider()

    if provider == "openai":
        return _create_openai_embedder()
    elif provider == "kusto":
        return _create_kusto_embedder()
    else:  # lmstudio (default)
        return _create_lmstudio_embedder()


def _create_openai_embedder() -> Callable[[str], tuple[list[float], str]]:
    """Create OpenAI embedding function."""
    from langchain_openai import OpenAIEmbeddings

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set when EMBEDDING_PROVIDER=openai")

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=model, api_key=api_key)  # type: ignore

    def embed(text: str) -> tuple[list[float], str]:
        vector = embeddings.embed_query(text)
        return vector, model

    return embed


def _create_lmstudio_embedder() -> Callable[[str], tuple[list[float], str]]:
    """Create LMStudio embedding function using direct HTTP client."""
    import json
    import urllib.request

    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    model = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

    # Remove trailing slash from base_url if present
    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/embeddings"

    def embed(text: str) -> tuple[list[float], str]:
        payload = {
            "model": model,
            "input": text,
        }

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                result = json.loads(response.read().decode("utf-8"))
                # Extract embedding from response: data[0].embedding
                vector = result["data"][0]["embedding"]
                return vector, model
        except Exception as e:
            raise RuntimeError(f"Failed to get embeddings from LMStudio at {endpoint}: {e}") from e

    return embed


def _create_kusto_embedder() -> Callable[[str], tuple[list[float], str]]:
    """Create Kusto ai_embeddings based embedding function."""
    from langgraph_kusto.common.kusto_client import KustoClient
    from langgraph_kusto.store.embeddings import KustoOpenAIEmbeddingFn

    model_uri = os.getenv("KUSTO_EMBEDDING_MODEL_URI")
    if not model_uri:
        raise RuntimeError("KUSTO_EMBEDDING_MODEL_URI must be set when EMBEDDING_PROVIDER=kusto")

    client = KustoClient.from_env()
    return KustoOpenAIEmbeddingFn(client=client, model_uri=model_uri)


def print_config():
    """Print current configuration for debugging."""
    print(f"LLM Provider: {get_llm_provider()}")
    print(f"Embedding Provider: {get_embedding_provider()}")
    if get_llm_provider() == "lmstudio":
        print(f"  LMStudio URL: {os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')}")
    if get_embedding_provider() == "lmstudio":
        print(f"  Embedding Model: {os.getenv('LMSTUDIO_EMBEDDING_MODEL', 'text-embedding-nomic-embed-text-v1.5')}")
