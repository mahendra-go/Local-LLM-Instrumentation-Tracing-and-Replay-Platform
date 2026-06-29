#!/usr/bin/env bash
# Downloads a small GGUF model for testing the capture backend.
# Usage: scripts/get-model.sh [tiny|qwen]
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

choice="${1:-qwen}"
case "$choice" in
  qwen)
    name="qwen2.5-0.5b-instruct-q4_k_m.gguf"
    url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/${name}"
    ;;
  tiny)
    name="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/${name}"
    ;;
  *)
    echo "unknown model '$choice' (use: qwen | tiny)"; exit 1 ;;
esac

dest="models/${name}"
if [[ -f "$dest" ]]; then
  echo "already present: $dest"
  exit 0
fi

echo "Downloading $name ..."
curl -L --fail --progress-bar -o "$dest" "$url"
echo "Saved -> $dest"
echo "Run: ./build/llmtrace --model $dest --prompt 'Explain attention briefly.'"
