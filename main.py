#!/usr/bin/env python3
"""
Krastler AI — main entry point.

Usage:
  python main.py                          # use config.yaml defaults
  python main.py --mode text              # keyboard-only (no mic/speaker)
  python main.py --mode audio             # full voice I/O
  python main.py --model mistral          # override Ollama model
  python main.py --config my_config.yaml  # custom config file
  python main.py --list-models            # print available Ollama models
"""

import argparse
import os
import sys

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[warn] Config file not found: {path} — using built-in defaults")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _print_banner(name: str, model: str, mode: str) -> None:
    width = 52
    print("=" * width)
    print(f"  {name} AI")
    print(f"  Model : {model}")
    print(f"  Mode  : {mode}")
    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Krastler AI — local personality powered by Ollama"
    )
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument("--mode", choices=["text", "audio", "hybrid"],
                   help="Interaction mode (overrides config)")
    p.add_argument("--model", help="Ollama model name (overrides config)")
    p.add_argument("--personality", help="Path to a personality YAML profile")
    p.add_argument("--list-models", action="store_true",
                   help="List available Ollama models and exit")
    p.add_argument("--no-audio", action="store_true",
                   help="Disable audio even if enabled in config")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------------------------

def run(config: dict, args: argparse.Namespace) -> None:
    # --- imports after config is loaded so logging is set up first ---
    from utils import logger as log_util

    log_level = _deep_get(config, "logging", "level", default="INFO")
    log_util.setup(log_level)

    import logging
    logger = logging.getLogger("krastler")

    from ai.conversation import ConversationHistory
    from ai.ollama_client import OllamaClient
    from personality.personality import Personality

    # --- Personality ---
    profile_path = (
        args.personality
        or _deep_get(config, "personality", "profile")
        or "personality/profiles/default.yaml"
    )
    personality = Personality(profile_path)

    # --- Ollama ---
    ollama_cfg = config.get("ollama", {})
    client = OllamaClient(
        host=ollama_cfg.get("host", "localhost"),
        port=int(ollama_cfg.get("port", 11434)),
    )
    model = args.model or ollama_cfg.get("model", "llama3.2")
    temperature = float(ollama_cfg.get("temperature", 0.7))

    if args.list_models:
        models = client.list_models()
        if models:
            print("Available Ollama models:")
            for m in models:
                print(f"  {m}")
        else:
            print("No models found (is Ollama running?)")
        return

    if not client.is_available():
        print(
            "[error] Cannot reach Ollama. Make sure it is running:\n"
            "  ollama serve\n"
            "Then pull a model if you haven't already:\n"
            f"  ollama pull {model}"
        )
        sys.exit(1)

    # --- Conversation history ---
    conv_cfg = config.get("conversation", {})
    history = ConversationHistory(
        max_turns=conv_cfg.get("max_turns", 20),
        system_prompt=personality.system_prompt,
    )
    history_path = conv_cfg.get("history_path", ".krastler_history.json")
    if conv_cfg.get("save_history", False):
        history.load(history_path)

    # --- Audio ---
    audio_cfg = config.get("audio", {})
    mode = args.mode or _deep_get(config, "interaction", "mode", default="hybrid")
    if args.no_audio:
        mode = "text"

    listener = None
    speaker = None
    audio_enabled = audio_cfg.get("enabled", True) and mode in ("audio", "hybrid")

    if audio_enabled:
        from audio.listener import AudioListener
        from audio.speaker import Speaker

        logger.info("Initialising audio (this may take a moment for model loading) ...")
        listener = AudioListener(audio_cfg)
        # Merge top-level audio keys (output_device, device_sample_rate, etc.)
        # with the tts sub-config so Speaker receives both.
        tts_cfg = {**audio_cfg, **audio_cfg.get("tts", {})}
        speaker = Speaker(tts_cfg)

        if not listener.is_ready:
            logger.warning("STT not ready — falling back to text input")
            listener = None
        if not speaker.is_ready:
            logger.warning("TTS not ready — responses will be printed only")
            speaker = None

        if mode == "audio" and listener is None:
            print("[error] Audio mode requested but microphone/STT is not available.")
            sys.exit(1)

    # --- Vision (optional) ---
    vision_cfg = config.get("vision", {})
    if vision_cfg.get("enabled", False):
        from core.event_bus import EventBus
        from core.orchestrator import Orchestrator
        from vision.camera import CameraModule

        orch = Orchestrator(config)
        cam = CameraModule()
        orch.register(cam, vision_cfg)
        orch.start()
    else:
        orch = None

    _print_banner(personality.name, model, mode)

    # --- Determine effective input mode ---
    use_voice = listener is not None
    if mode == "text":
        use_voice = False

    _print_help(personality.name, use_voice)

    # --- Main loop ---
    try:
        while True:
            user_input = _get_input(personality.name, use_voice, listener, speaker)
            if user_input is None:
                continue

            cmd = user_input.strip().lower()
            if cmd in ("exit", "quit", "bye"):
                print(f"\n{personality.name}: Goodbye!\n")
                if speaker:
                    speaker.speak("Goodbye!")
                break
            if cmd == "clear":
                history.clear()
                print("  [Conversation history cleared]\n")
                continue
            if cmd == "help":
                _print_help(personality.name, use_voice)
                continue
            if cmd.startswith("!mode "):
                new_mode = cmd.split(" ", 1)[1].strip()
                if new_mode in ("text", "voice"):
                    use_voice = new_mode == "voice" and listener is not None
                    print(f"  [Switched to {new_mode} mode]\n")
                continue

            # Build and send to Ollama
            history.add_user(user_input)
            print(f"\n{personality.name}: ", end="", flush=True)

            messages = history.get_messages()
            token_gen = client.chat_stream(model, messages, temperature)

            if speaker:
                full_response = speaker.speak_stream(_tee_print(token_gen))
            else:
                full_response = _print_stream(token_gen)

            print("\n")
            history.add_assistant(full_response)

            if conv_cfg.get("save_history", False):
                history.save(history_path)

    except KeyboardInterrupt:
        print(f"\n\n{personality.name}: Shutting down. Goodbye!")

    finally:
        if orch:
            orch.stop()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _get_input(name: str, use_voice: bool, listener, speaker) -> str | None:
    """Get input from mic or keyboard depending on mode."""
    if use_voice:
        print(f"[Listening...] (Press Ctrl+C to stop, or type below and press Enter)")
        # Run voice and text input concurrently using a simple thread race
        import queue
        import threading

        result_q: queue.Queue[str] = queue.Queue()

        def voice_thread():
            text = listener.listen()
            if text:
                result_q.put(text)

        def text_thread():
            try:
                text = input()
                if text.strip():
                    result_q.put(text.strip())
            except EOFError:
                pass

        vt = threading.Thread(target=voice_thread, daemon=True)
        tt = threading.Thread(target=text_thread, daemon=True)
        vt.start()
        tt.start()

        user_input = result_q.get()  # wait for whichever arrives first
        print(f"You: {user_input}")
        return user_input
    else:
        try:
            user_input = input(f"You: ").strip()
            return user_input if user_input else None
        except EOFError:
            return "exit"


def _tee_print(gen):
    """Yield tokens from gen while also printing them to stdout."""
    for token in gen:
        print(token, end="", flush=True)
        yield token


def _print_stream(gen) -> str:
    """Print streaming tokens and return the full assembled text."""
    parts = []
    for token in gen:
        print(token, end="", flush=True)
        parts.append(token)
    return "".join(parts)


def _print_help(name: str, use_voice: bool) -> None:
    mode_str = "voice + keyboard" if use_voice else "keyboard"
    print(f"Input mode  : {mode_str}")
    print("Commands    : exit | clear | help | !mode text | !mode voice")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    config = _load_config(args.config)
    run(config, args)
