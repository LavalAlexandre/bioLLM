#!/bin/bash

echo "🔄 Restarting vLLM server..."

# Stop all vLLM containers
echo "🛑 Stopping vLLM containers..."
docker stop $(docker ps -q --filter ancestor=vllm/vllm-openai:latest) 2>/dev/null || echo "No running containers found"

# Remove stopped containers to ensure clean start
echo "🗑️  Removing old containers..."
docker rm $(docker ps -aq --filter ancestor=vllm/vllm-openai:latest) 2>/dev/null || echo "No containers to remove"

# Wait a moment for cleanup
sleep 2

# Start fresh vLLM server
echo "🚀 Starting vLLM server..."
./start_vllm_docker-8B.sh

echo "✅ vLLM restart complete!"
echo "⏳ Wait ~30 seconds for model to load..."
