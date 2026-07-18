import os

DEFAULT_BASE_URL = "http://localhost:8000/v1"
SECRET_NAME = "GEMMA_VLLM_BASE_URL"


def resolve_base_url(ctx):
    """Resolves the vLLM server's OpenAI-compatible base URL.

    Checks the plugin secret first (set via fiftyone.yml `secrets:` +
    the GEMMA_VLLM_BASE_URL env var), falling back to the default
    localhost address vLLM's docker compose serves on.
    """
    secret = ctx.secrets.get(SECRET_NAME) if ctx.secrets else None
    return secret or os.environ.get(SECRET_NAME) or DEFAULT_BASE_URL


def metrics_url(base_url):
    """vLLM exposes /metrics on the same host:port as /v1, not under /v1."""
    root = base_url.rsplit("/v1", 1)[0].rstrip("/")
    return f"{root}/metrics"


def models_url(base_url):
    return f"{base_url.rstrip('/')}/models"
