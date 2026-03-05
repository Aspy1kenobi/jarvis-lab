"""
Async HTTP client for communicating with a local Ollama instance.
"""

import httpx
from config import Config


async def call_ollama(
    messages: list[dict],
    config: Config,
) -> tuple[str, dict]:
    """
    POST a chat request to the Ollama /api/chat endpoint.

    Args:
        messages: List of message dicts, e.g. [{"role": "user", "content": "..."}]
        config:   Config instance carrying model name, base URL, and generation params.

    Returns:
        A (text, usage) tuple where:
          - text  is the assistant's reply string
          - usage is a dict containing at minimum {"eval_count": N}

    Raises:
        httpx.HTTPStatusError: if the server returns a non-200 status.
        httpx.TimeoutException: if the request exceeds config.timeout seconds.
    """
    url = f"{config.api_base_url}/api/chat"

    payload = {
        "model": config.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        response = await client.post(url, json=payload)

        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Ollama returned {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )

        data = response.json()

    text: str = data["message"]["content"]

    usage: dict = {
        "eval_count": data.get("eval_count", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
    }

    return text, usage