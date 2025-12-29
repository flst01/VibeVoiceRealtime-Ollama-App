# VibeVoiceRealtime-Ollama-App

This project is an adapted demo built on top of the original [VibeVoice](https://github.com/microsoft/VibeVoice) work. Huge thanks to the original project team for their great work and open release.

## What this demo does

VibeVoiceRealtime-Ollama-App shows a direct implementation of streaming a **local Ollama LLM** response into a **real-time TTS** web demo. The browser streams text deltas to the UI while the server sentence-buffers the stream and feeds segments to the existing VibeVoice realtime TTS pipeline so audio can start before the LLM finishes.

## Quick start

```bash
# Start Ollama and pull a model (example)
ollama serve
ollama pull llama3.2

# Launch the realtime demo
OLLAMA_BASE_URL=http://localhost:11434 \
OLLAMA_MODEL=llama3.2 \
python demo/vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B
```

Optional: switch to the Responses API mode.

```bash
OLLAMA_API_MODE=responses \
python demo/vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B
```

Smoke test for the Ollama streaming endpoint:

```bash
bash demo/web/smoke_ollama_stream.sh "Give me a short story about trees."
```

## Credits

- Original project: [VibeVoice](https://github.com/microsoft/VibeVoice)
- Most changes to the original repo were made with [OpenAI Codex](https://github.com/openai/codex)
