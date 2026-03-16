import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE = os.path.join(
    os.path.dirname(__file__), "profiles", "default.yaml"
)


class Personality:
    """
    Loads a YAML personality profile and builds the system prompt used for
    every conversation with the LLM.
    """

    def __init__(self, profile_path: str | None = None):
        self._profile: dict[str, Any] = {}
        self._system_prompt: str = ""

        path = profile_path or _DEFAULT_PROFILE
        if os.path.exists(path):
            self._load(path)
        else:
            logger.warning("Personality profile not found at '%s', using built-in defaults", path)
            self._profile = {
                "name": "Krastler",
                "description": "a helpful AI companion",
                "traits": ["curious", "helpful", "direct"],
            }
            self._build_prompt()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._profile.get("name", "Krastler")

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def reload(self, profile_path: str) -> None:
        self._load(profile_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self, filepath: str) -> None:
        with open(filepath) as f:
            self._profile = yaml.safe_load(f) or {}
        logger.info("Loaded personality profile: %s", self._profile.get("name", "Unknown"))
        self._build_prompt()

    def _build_prompt(self) -> None:
        # If the profile provides a hand-crafted system prompt, use it verbatim.
        if raw := self._profile.get("system_prompt"):
            self._system_prompt = raw.strip()
            return

        name = self._profile.get("name", "AI")
        description = self._profile.get("description", "a helpful AI assistant")
        traits: list[str] = self._profile.get("traits", [])
        speaking_style: str = self._profile.get("speaking_style", "conversational")
        knowledge_areas: list[str] = self._profile.get("knowledge_areas", [])
        guidelines: list[str] = self._profile.get("guidelines", [])

        parts: list[str] = [f"You are {name}, {description}."]

        if traits:
            readable = [t.replace("_", " ") for t in traits]
            parts.append(f"Your core personality traits are: {', '.join(readable)}.")

        if speaking_style:
            parts.append(f"Speaking style: {speaking_style}")

        if knowledge_areas:
            parts.append(f"You have deep knowledge in: {', '.join(knowledge_areas)}.")

        if guidelines:
            parts.append("\nGuidelines:")
            for g in guidelines:
                parts.append(f"- {g}")

        self._system_prompt = "\n".join(parts)
        logger.debug("Built system prompt (%d chars)", len(self._system_prompt))
