#!/bin/bash

# Define the cache directory on the large EBS volume
CACHE_DIR="/home/ec2-user/SageMaker/hf_cache"
# Ensure the cache directory exists
mkdir -p "$CACHE_DIR"

# Start vLLM Docker container with OpenAI-compatible server
# Based on: https://docs.vllm.ai/en/stable/getting_started/quickstart.html#openai-compatible-server
# The HF Token is only necessary if you are using a private model or one with a required user agreement
docker run --runtime nvidia --gpus '"device=0,1,2,3"' \
    -v "$CACHE_DIR":/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --env "HF_HOME=/home/ec2-user/SageMaker/.cache" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model openai/gpt-oss-20b \
    --tensor-parallel-size 4 \
    --port 8000 \