import json
import logging
from typing import Generator

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """HTTP client for the Ollama API."""

    def __init__(self, host: str = "localhost", port: int = 11434):
        self.base_url = f"http://{host}:{port}"

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as e:
            logger.error("Failed to list models: %s", e)
            return []

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Stream chat completion tokens. Yields text fragments."""
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": True,
        }
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            yield "I can't reach my brain right now — Ollama doesn't appear to be running."
        except Exception as e:
            logger.exception("Chat stream error")
            yield f"Something went wrong: {e}"

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        """Non-streaming chat — returns the full response string."""
        return "".join(self.chat_stream(model, messages, temperature))
