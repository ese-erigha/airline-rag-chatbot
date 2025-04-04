#!/bin/bash

# Start Ollama server in the background
ollama serve &

# Wait for Ollama server to start
sleep 5

# Pull llama3 model and mxbai-embed-large embedding model
#ollama pull llama3.2; ollama pull mxbai-embed-large;

MODELS=("llama3.2" "mxbai-embed-large")
for MODEL in "${MODELS[@]}"; do
  echo "Downloading model: $MODEL"
  while ! ollama pull "$MODEL"; do
    echo "Download failed for $MODEL. Retrying..."
    sleep 0.1
  done
  echo "Download successful for $MODEL."
done


# Wait for the Ollama server to finish
wait $!
