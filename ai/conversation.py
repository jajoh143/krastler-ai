import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class ConversationHistory:
    """
    Manages a rolling window of conversation messages.

    Each entry keeps role + content; timestamps are stored internally but
    stripped when building the message list sent to Ollama.
    """

    def __init__(self, max_turns: int = 20, system_prompt: str = ""):
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self._messages: list[dict] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_user(self, content: str) -> None:
        self._append("user", content)

    def add_assistant(self, content: str) -> None:
        self._append("assistant", content)

    def clear(self) -> None:
        self._messages = []
        logger.info("Conversation history cleared")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_messages(self) -> list[dict]:
        """Return the message list in the format expected by Ollama's /api/chat."""
        out: list[dict] = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        for msg in self._messages:
            out.append({"role": msg["role"], "content": msg["content"]})
        return out

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self._messages, f, indent=2)
        logger.debug("Saved conversation to %s", filepath)

    def load(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            return
        with open(filepath) as f:
            self._messages = json.load(f)
        logger.info("Loaded %d messages from %s", len(self._messages), filepath)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(self, role: str, content: str) -> None:
        self._messages.append(
            {
                "role": role,
                "content": content,
                "ts": datetime.now().isoformat(),
            }
        )
        self._trim()

    def _trim(self) -> None:
        # Keep at most max_turns user+assistant pairs (2 messages per turn)
        limit = self.max_turns * 2
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]

    def __len__(self) -> int:
        return len(self._messages)
