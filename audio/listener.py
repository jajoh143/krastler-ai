"""
Audio listener module — captures microphone input and transcribes speech.

STT engines supported:
  - faster-whisper  (default, best accuracy, runs offline on RPi 5)
  - vosk            (lighter alternative, also offline)

Voice activity detection (VAD) is done via energy threshold + silence timer,
keeping dependencies minimal and compatible with RPi.
"""

import logging
import threading
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioListener:
    """
    Records audio from the microphone until a configurable silence gap is
    detected, then hands the captured audio to a transcription engine.

    Usage:
        listener = AudioListener(config)
        text = listener.listen()   # blocks until speech + silence detected
    """

    def __init__(self, config: dict):
        self.sample_rate: int = config.get("sample_rate", 16000)
        self.chunk_size: int = config.get("chunk_size", 1024)
        self.silence_threshold: float = config.get("silence_threshold", 0.02)
        self.silence_duration: float = config.get("silence_duration", 1.5)
        self.min_speech_duration: float = config.get("min_speech_duration", 0.4)
        self.max_record_seconds: float = config.get("max_record_seconds", 30.0)
        self.input_device: Optional[int] = config.get("input_device", None)

        self._stt_engine = None
        self._stt_type: Optional[str] = None

        stt_cfg: dict = config.get("stt", {})
        self._language: str = stt_cfg.get("language", "en")
        self._init_stt(stt_cfg.get("engine", "faster-whisper"), stt_cfg)

    # ------------------------------------------------------------------
    # STT initialisation
    # ------------------------------------------------------------------

    def _init_stt(self, engine: str, cfg: dict) -> None:
        if engine == "faster-whisper":
            self._init_faster_whisper(cfg)
        elif engine == "vosk":
            self._init_vosk(cfg)
        else:
            logger.error("Unknown STT engine: %s", engine)

        if self._stt_engine is None and engine != "vosk":
            logger.warning("Falling back to vosk")
            self._init_vosk(cfg)

    def _init_faster_whisper(self, cfg: dict) -> None:
        try:
            from faster_whisper import WhisperModel

            model_size = cfg.get("model", "base")
            compute_type = cfg.get("compute_type", "int8")
            logger.info("Loading Whisper model '%s' (compute_type=%s) ...", model_size, compute_type)
            self._stt_engine = WhisperModel(
                model_size, device="cpu", compute_type=compute_type
            )
            self._stt_type = "faster-whisper"
            logger.info("Whisper model ready")
        except ImportError:
            logger.error("faster-whisper not installed — run: pip install faster-whisper")
        except Exception:
            logger.exception("Failed to load Whisper model")

    def _init_vosk(self, cfg: dict) -> None:
        try:
            from vosk import KaldiRecognizer, Model

            model_path = cfg.get("model_path", "models/vosk-model-small-en")
            logger.info("Loading Vosk model from '%s' ...", model_path)
            self._vosk_model = Model(model_path)
            self._stt_engine = KaldiRecognizer(self._vosk_model, self.sample_rate)
            self._stt_type = "vosk"
            logger.info("Vosk model ready")
        except ImportError:
            logger.error("vosk not installed — run: pip install vosk")
        except Exception:
            logger.exception("Failed to load Vosk model")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def listen(self) -> Optional[str]:
        """
        Block until speech is detected, then record until silence.
        Returns the transcribed text, or None if nothing was captured.
        """
        import sounddevice as sd

        audio_chunks: list[np.ndarray] = []
        stop_event = threading.Event()
        silent_chunks = 0
        speech_chunks = 0

        silence_limit = int(self.silence_duration * self.sample_rate / self.chunk_size)
        min_speech = int(self.min_speech_duration * self.sample_rate / self.chunk_size)

        def callback(indata: np.ndarray, frames: int, time, status) -> None:
            nonlocal silent_chunks, speech_chunks
            rms = float(np.sqrt(np.mean(indata ** 2)))

            if rms >= self.silence_threshold:
                silent_chunks = 0
                speech_chunks += 1
                audio_chunks.append(indata.copy())
            elif speech_chunks > 0:
                silent_chunks += 1
                audio_chunks.append(indata.copy())
                if silent_chunks >= silence_limit:
                    stop_event.set()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.chunk_size,
            device=self.input_device,
            callback=callback,
        ):
            stop_event.wait(timeout=self.max_record_seconds)

        if speech_chunks < min_speech or not audio_chunks:
            return None

        audio = np.concatenate(audio_chunks).flatten()
        return self._transcribe(audio)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        if self._stt_type == "faster-whisper":
            return self._transcribe_whisper(audio)
        if self._stt_type == "vosk":
            return self._transcribe_vosk(audio)
        logger.error("No STT engine available")
        return None

    def _transcribe_whisper(self, audio: np.ndarray) -> Optional[str]:
        segments, _ = self._stt_engine.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text for seg in segments).strip()
        logger.debug("Whisper transcript: %r", text)
        return text or None

    def _transcribe_vosk(self, audio: np.ndarray) -> Optional[str]:
        import json

        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        if self._stt_engine.AcceptWaveform(audio_bytes):
            result = json.loads(self._stt_engine.Result())
            text = result.get("text", "").strip()
        else:
            result = json.loads(self._stt_engine.PartialResult())
            text = result.get("partial", "").strip()
        logger.debug("Vosk transcript: %r", text)
        return text or None

    @property
    def is_ready(self) -> bool:
        return self._stt_engine is not None
