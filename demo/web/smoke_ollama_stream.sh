#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${OLLAMA_BASE_URL:-http://localhost:11434}
MODEL=${OLLAMA_MODEL:-llama3.2}
PROMPT=${1:-"Give me a quick overview of realtime text-to-speech."}

curl -N -sS "${BASE_URL%/}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}]}"
