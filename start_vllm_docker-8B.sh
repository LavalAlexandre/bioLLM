#!/bin/bash

# Define the cache directory on the large EBS volume
CACHE_DIR="/home/ec2-user/SageMaker/data/hf_cache"
# Ensure the cache directory exists
mkdir -p "$CACHE_DIR"

# Start vLLM Docker container with OpenAI-compatible server
# Based on: https://docs.vllm.ai/en/stable/getting_started/quickstart.html#openai-compatible-server
# The HF Token is only necessary if you are using a private model or one with a required user agreement
#
# OPTIMIZATIONS FOR MAXIMUM SPEED:
# --tensor-parallel-size 8: Use ALL 8 GPUs (FIXED from 4!)
# --enable-auto-tool-choice: Enable tool calling
# --tool-call-parser qwen: Use NATIVE Qwen format (CRITICAL FIX - was hermes!)
# --enable-prefix-caching: Cache tool descriptions (30-50% speedup!)
# --max-num-seqs 256: Allow up to 256 concurrent sequences
# --gpu-memory-utilization 0.90: Use 90% of GPU memory
# --disable-log-requests: Reduce logging overhead

docker run --runtime nvidia --gpus '"device=0,1,2,3,4,5,6,7"' \
    -v "$CACHE_DIR":/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --env "HF_HOME=/home/ec2-user/SageMaker/.cache" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-8B \
    --tensor-parallel-size 8 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen \
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 \
    --disable-log-requests
