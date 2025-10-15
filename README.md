# bioLLM - Biological Question Answering System

A high-performance biological question answering system using vLLM inference with integrated cBioPortal and protein expression tools.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
bash install.sh

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 2. Start vLLM Server

Start the vLLM inference server (requires GPU):

```bash
# For Qwen 8B model
bash start_vllm_docker-8B.sh

# Or for GPT-OSS model
bash start_vllm_docker-gpt-oss.sh
```

The server will be available at `http://localhost:8000`

### 3. Run Inference

```bash
python main.py
```

This will:
- Load test questions from `data/hackathon-test.jsonl`
- Process them using the agent with tool calls
- Save results to `result/test_answers.jsonl`

## 📁 Project Structure

```
bioLLM/
├── main.py                          # Entry point
├── src/
│   ├── classify.py                  # Question classification and processing
│   ├── data_preprocess.py           # Data loading and prompt formatting
│   └── model/
│       ├── model.py                 # vLLM model wrapper with agent
│       ├── cbioportal_tool.py       # cBioPortal cancer genomics tool
│       ├── protein_expression_tool.py # Protein expression tool
│       ├── biorxiv_tool.py          # BioRxiv literature search
│       └── cbioportal/              # Modular cBioPortal client
│           ├── client.py            # API client with connection pooling
│           ├── mutations.py         # Mutation data fetching
│           ├── expression.py        # Expression data fetching
│           ├── copy_number.py       # CNA data fetching
│           ├── clinical.py          # Clinical data fetching
│           └── utils.py             # Helper functions
├── data/
│   ├── hackathon-train.jsonl        # Training questions
│   └── hackathon-test.jsonl         # Test questions
└── result/
    └── test_answers.jsonl           # Generated answers
```

## 🛠️ Configuration

### Model Settings

Edit `src/model/model.py` to configure:

```python
Model(
    temperature=0.6,                    # Sampling temperature
    enable_cbioportal=True,             # Enable cBioPortal tool
    enable_protein_expression=True,     # Enable protein expression tool
    enable_biorxiv=False,              # Enable BioRxiv search
    max_concurrent=80,                  # Concurrent requests (auto-tuned for 8 GPUs)
    request_timeout=400.0              # Timeout per request (seconds)
)
```

### Batch Processing

Edit `src/classify.py` to configure batching:

```python
BATCH_SIZE = 32                        # Questions per batch
MAX_TOKENS_PER_QUESTION = 2048        # Max tokens per answer
TEMPERATURE = 0.6                      # Generation temperature
```

## 🔧 Features

### Async Tool Execution
- Non-blocking I/O for cBioPortal API calls
- Thread pool executors prevent event loop blocking
- Optimized connection pooling (100 connections)

### vLLM Optimizations
- Continuous batching for optimal GPU utilization
- Prefix caching enabled (90%+ hit rate)
- Concurrent request processing (80 concurrent by default)
- Hermes tool call parser for structured output

### Integrated Tools

#### 1. cBioPortal Tool (`search_cbioportal`)
Fetches real-world cancer genomics data:
- Mutation frequencies and hotspots
- mRNA expression (tumor and normal tissue)
- Protein expression (RPPA)
- Copy number alterations (CNA)
- Clinical and survival data

#### 2. Protein Expression Tool (`search_protein_expression`)
Wrapper around cBioPortal for protein-focused queries.

#### 3. BioRxiv Tool (optional)
Search biological literature (disabled by default for performance).

## 📊 Data Format

### Input Format (`hackathon-test.jsonl`)
```json
{
  "question": "Which gene is most frequently mutated in lung adenocarcinoma?",
  "options": "{\"A\": \"EGFR\", \"B\": \"KRAS\", \"C\": \"TP53\", \"D\": \"ALK\"}",
  "question_type": "mutation_frequency",
  "metadata": "...",
  "dataset_name": "TCGA Mutations"
}
```

### Output Format (`result/test_answers.jsonl`)
```json
{
  "question": "Which gene is most frequently mutated in lung adenocarcinoma?",
  "options": "{\"A\": \"EGFR\", \"B\": \"KRAS\", \"C\": \"TP53\", \"D\": \"ALK\"}",
  "answer_letter": "C",
  "raw_response": "<think>Using cBioPortal data...</think><answer>C</answer>",
  "question_type": "mutation_frequency",
  "metadata": "...",
  "dataset_name": "TCGA Mutations"
}
```

## ⚡ Performance Optimization

### GPU Utilization
- **8 GPUs**: Recommended concurrency = 80 (10x per GPU)
- **4 GPUs**: Recommended concurrency = 40
- **Adjust in `model.py`**: `max_concurrent` parameter

### Throughput Tuning
- **Batch size**: Larger = higher throughput, more memory
- **Temperature**: Lower = faster (less sampling)
- **Max tokens**: Lower = faster generation

### Monitoring
Check vLLM logs for performance metrics:
```bash
docker logs -f <container_id>
```

Look for:
- `Avg prompt throughput: X tokens/s`
- `Avg generation throughput: X tokens/s`
- `GPU KV cache usage: X%`
- `Prefix cache hit rate: X%`

## 🐛 Troubleshooting

### vLLM Server Issues

```bash
# Check if server is running
curl http://localhost:8000/v1/models

# View logs
docker ps  # Get container ID
docker logs <container_id>

# Restart server
docker stop <container_id>
bash start_vllm_docker-8B.sh
```

### GPU Memory Issues

```bash
# Check GPU usage
nvidia-smi

# Kill stuck processes
nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9
```

### Connection Pool Warnings

If you see "Connection pool is full" warnings, increase pool size in `src/model/cbioportal/client.py`:

```python
pool_maxsize=200,  # Increase from 100
```

## 📝 Development

### Adding New Tools

1. Create tool function with `@function_tool` decorator:
```python
from agents import function_tool

@function_tool
async def my_new_tool(query: str) -> str:
    """Tool description for the LLM."""
    # Implementation
    return result
```

2. Add to agent in `src/model/model.py`:
```python
tools.append(my_new_tool)
```

### Prompt Engineering

Edit agent instructions in `src/model/model.py`:
```python
instructions="""Your custom instructions here..."""
```

## 📚 Dependencies

Key packages:
- `vllm` - Fast LLM inference
- `agents` - Agent framework with tool calling
- `bravado` - cBioPortal API client
- `httpx` - Async HTTP client
- `openai` - OpenAI-compatible API

See `pyproject.toml` for full list.

## 🔒 License

See repository license.

## 🤝 Contributing

This is a production-ready system. For improvements, consider:
- Adding comprehensive error handling
- Implementing request rate limiting
- Adding authentication for API endpoints
- Setting up monitoring and logging infrastructure
- Adding unit and integration tests
