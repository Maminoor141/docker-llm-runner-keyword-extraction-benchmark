#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Start Ollama serve in the background
ollama serve &

# Get the process ID of the server
pid=$!

# Wait for the server to be available
echo "Waiting for Ollama server to start..."
while ! ollama list > /dev/null 2>&1
do
  sleep 1
done
echo "Ollama server started."

# Check if the model is already available to avoid re-downloading
echo "Checking for model: mistral:latest"
if ! ollama list | grep -q 'mistral:latest'; then
  echo "Model not found locally. Pulling..."
  ollama pull mistral:latest
  echo "Model pull complete."
else
  echo "Model 'mistral:latest' already exists."
fi

echo "Ollama is fully ready and serving requests."

# Wait for the background server process. This will keep the container running.
wait $pid