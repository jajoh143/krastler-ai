# Krastler AI

A modular, local AI personality running on a Raspberry Pi via [Ollama](https://ollama.com).
Krastler listens, thinks, and speaks — entirely offline.

## Features

| Module | Status | Description |
|---|---|---|
| AI personality | ✅ Active | YAML-configurable character, traits, system prompt |
| Ollama client | ✅ Active | Streaming chat with any Ollama model |
| Speech-to-Text | ✅ Active | `faster-whisper` (offline, RPi 5 optimised) |
| Text-to-Speech | ✅ Active | `piper-tts` (natural voice, offline) |
| Vision / Camera | 🔧 Scaffold | Frame capture stub, ready for face/object detection |
| Autonomous driving | 📋 Planned | GPIO motor controller (modular, not yet implemented) |
| Facial recognition | 📋 Planned | `face_recognition` integration (not yet implemented) |

---

## Requirements

### Hardware
- Raspberry Pi 5 (recommended) or Pi 4 (4 GB+)
- USB microphone or USB audio adapter with mic
- Speaker (3.5 mm or USB audio)
- (Optional) USB/CSI camera

### System packages
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev libespeak-ng1 espeak-ng ffmpeg python3-pip
```

### Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2     # or whichever model you prefer
ollama serve             # keep this running in a separate terminal
```

---

## Setup

```bash
git clone <repo-url>
cd krastler-ai

# Install Python dependencies
pip install -r requirements.txt

# Download a Piper TTS voice model (~65 MB)
python setup_models.py

# (Optional) list other available voices
python setup_models.py --list
```

---

## Running

```bash
# Default (hybrid mode — voice if mic available, else text)
python main.py

# Text-only (no microphone needed — great for initial testing)
python main.py --mode text

# Full voice I/O
python main.py --mode audio

# Use a different Ollama model
python main.py --model mistral

# List your available Ollama models
python main.py --list-models
```

### In-conversation commands

| Command | Effect |
|---|---|
| `exit` / `quit` / `bye` | Shut down gracefully |
| `clear` | Clear conversation history |
| `help` | Show mode and commands |
| `!mode text` | Switch to keyboard input |
| `!mode voice` | Switch to voice input |

---

## Configuration

All settings live in `config.yaml`. Key options:

```yaml
ollama:
  model: "llama3.2"       # change to any model you have pulled

audio:
  stt:
    model: "base"         # tiny | base | small  (base recommended for RPi 5)
  tts:
    model_path: "models/piper/en_US-lessac-medium.onnx"

interaction:
  mode: "hybrid"          # text | audio | hybrid
```

### Customising the personality

Edit `personality/profiles/default.yaml` — or create a new profile and point
to it with `--personality my_profile.yaml` or in `config.yaml`.

```yaml
name: "Krastler"
traits: [curious, witty, helpful]
speaking_style: "conversational and clear"
guidelines:
  - "Keep answers concise unless asked for more detail."
# Or override everything with a fully hand-crafted system prompt:
# system_prompt: |
#   You are ...
```

---

## Architecture

```
krastler-ai/
├── main.py                    Entry point & conversation loop
├── config.yaml                All configuration
├── setup_models.py            Piper TTS model downloader
│
├── core/
│   ├── module_base.py         Abstract base for all modules
│   ├── event_bus.py           Thread-safe pub/sub event bus
│   └── orchestrator.py        Module coordinator
│
├── personality/
│   ├── personality.py         Loads profile → builds system prompt
│   └── profiles/
│       └── default.yaml       Default Krastler personality
│
├── ai/
│   ├── ollama_client.py       Streaming HTTP client for Ollama
│   └── conversation.py        Rolling conversation history
│
├── audio/
│   ├── listener.py            Mic capture + STT (faster-whisper / vosk)
│   └── speaker.py             TTS playback (piper / pyttsx3 fallback)
│
└── vision/
    └── camera.py              Camera frame capture (scaffold for future CV)
```

### Event bus

Modules communicate via `EventBus` events — keeping them fully decoupled:

| Event | Source | Consumers |
|---|---|---|
| `speech.detected` | listener | main loop |
| `ai.response_token` | main loop | speaker |
| `vision.frame` | camera | future CV modules |
| `vision.face_detected` | future | personality / driving |
| `drive.command` | future | motor controller |

### Adding a new module

```python
from core.module_base import Module
from core.event_bus import EventBus

class MyModule(Module):
    def __init__(self):
        super().__init__("my_module")

    def _setup(self):
        self.subscribe(EventBus.SPEECH_DETECTED, self._on_speech)

    def start(self):
        pass  # start background threads here

    def stop(self):
        pass  # clean up resources

    def _on_speech(self, text: str):
        print(f"Heard: {text}")
```

Then register it in `main.py`:
```python
orch.register(MyModule(), config.get("my_module", {}))
```

---

## Roadmap

- [ ] **Autonomous driving** — GPIO motor controller module with voice/LLM commands
- [ ] **Facial recognition** — recognise known faces, personalise responses
- [ ] **Wake word** — "Hey Krastler" to activate listening (Porcupine or Snowboy)
- [ ] **Display output** — show responses on a small HDMI/SPI display
- [ ] **Memory** — long-term memory using a local vector store (e.g. ChromaDB)
