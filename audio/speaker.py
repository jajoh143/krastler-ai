"""
Speaker module — converts text to speech and plays it back.

TTS engines supported:
  - piper   (default, natural-sounding, offline, great for RPi 5)
  - pyttsx3 (fallback, uses espeak, very lightweight)

For Piper, download a voice model first:
  python setup_models.py
"""

import io
import logging
import os
import re
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sentence-boundary pattern used for streaming TTS
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Speaker:
    """
    Synthesises speech from text and plays it via sounddevice.

    speak(text)         — synthesise and play synchronously
    speak_stream(gen)   — consume a token generator, speak sentence by sentence
                          as text arrives (lower perceived latency)
    """

    def __init__(self, config: dict):
        self.engine: str = config.get("engine", "piper")
        self.volume: float = config.get("volume", 1.0)
        self.output_device: Optional[int] = config.get("output_device", None)
        self._piper_voice = None
        self._pyttsx_engine = None

        if self.engine == "piper":
            self._init_piper(config)
            if self._piper_voice is None:
                logger.warning("Piper unavailable — falling back to pyttsx3")
                self.engine = "pyttsx3"
                self._init_pyttsx3(config)
        elif self.engine == "pyttsx3":
            self._init_pyttsx3(config)
        else:
            logger.error("Unknown TTS engine: %s", self.engine)

    # ------------------------------------------------------------------
    # Engine initialisation
    # ------------------------------------------------------------------

    def _init_piper(self, cfg: dict) -> None:
        model_path = cfg.get("model_path") or self._find_piper_model()
        if not model_path:
            logger.warning(
                "No Piper model found. Run: python setup_models.py  "
                "(then set audio.tts.model_path in config.yaml)"
            )
            return
        try:
            from piper.voice import PiperVoice

            logger.info("Loading Piper model: %s", model_path)
            self._piper_voice = PiperVoice.load(model_path, use_cuda=False)
            logger.info("Piper TTS ready")
        except ImportError:
            logger.error("piper-tts not installed — run: pip install piper-tts")
        except Exception:
            logger.exception("Failed to load Piper model")

    def _find_piper_model(self) -> Optional[str]:
        import glob

        patterns = ["models/piper/*.onnx", "~/.local/share/piper/*.onnx"]
        for pattern in patterns:
            matches = glob.glob(os.path.expanduser(pattern))
            if matches:
                return matches[0]
        return None

    def _init_pyttsx3(self, cfg: dict) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", cfg.get("rate", 155))
            engine.setProperty("volume", self.volume)
            voice_index = cfg.get("voice_index", 0)
            voices = engine.getProperty("voices")
            if voices and voice_index < len(voices):
                engine.setProperty("voice", voices[voice_index].id)
            self._pyttsx_engine = engine
            logger.info("pyttsx3 TTS ready")
        except ImportError:
            logger.error("pyttsx3 not installed — run: pip install pyttsx3")
        except Exception:
            logger.exception("Failed to initialise pyttsx3")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Synthesise and play text synchronously."""
        text = text.strip()
        if not text:
            return
        if self.engine == "piper" and self._piper_voice:
            self._play_piper(text)
        elif self._pyttsx_engine:
            self._play_pyttsx3(text)
        else:
            print(f"  [TTS unavailable] {text}")

    def speak_stream(self, token_generator) -> str:
        """
        Consume an iterable of text tokens (e.g. from Ollama streaming).
        Speaks each complete sentence as soon as it arrives, minimising
        the gap between generation and playback.

        Returns the full assembled text.
        """
        buffer = ""
        full_text = ""
        for token in token_generator:
            buffer += token
            full_text += token
            # Flush complete sentences immediately
            parts = _SENTENCE_END.split(buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        self.speak(sentence)
                buffer = parts[-1]

        # Speak any remaining text
        if buffer.strip():
            self.speak(buffer.strip())

        return full_text

    @property
    def is_ready(self) -> bool:
        return self._piper_voice is not None or self._pyttsx_engine is not None

    # ------------------------------------------------------------------
    # Playback internals
    # ------------------------------------------------------------------

    def _play_piper(self, text: str) -> None:
        try:
            import sounddevice as sd

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                self._piper_voice.synthesize(text, wf)
            buf.seek(0)
            with wave.open(buf) as wf:
                framerate = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            audio = np.clip(audio * self.volume, -1.0, 1.0)
            sd.play(audio, samplerate=framerate, device=self.output_device)
            sd.wait()
        except Exception:
            logger.exception("Piper playback error")

    def _play_pyttsx3(self, text: str) -> None:
        try:
            self._pyttsx_engine.say(text)
            self._pyttsx_engine.runAndWait()
        except Exception:
            logger.exception("pyttsx3 playback error")
