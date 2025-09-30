#!/bin/bash

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

source $HOME/.local/bin/env

# Pull docker image for vllm
docker pull vllm/vllm-openai

# Create and sync virtual environment
uv venv --python python3.11
source .venv/bin/activate
uv sync
# Optional : install vllm in your virtual environment
# This is not necessary if you are using the vLLM Docker container
# CC=/usr/bin/gcc10-gcc CXX=/usr/bin/gcc10-g++ uv pip install vllm --no-build-isolation

# Install ipykernel
python -m ipykernel install \
    --user \
    --name="decoding-biology-hackathon-platform-kernel" \
    --display-name="Python (Decoding Biology Hackathon Platform)"

echo "Installed ipykernel, refresh the jupyter lab page to see the new kernel!"

echo "Installation complete. Activate with: source .venv/bin/activate"
