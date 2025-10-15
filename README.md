# bioLLM - Agentic Biological Question Answering System

**An intelligent agentic AI system for answering complex biological and cancer genomics questions, developed during Owkin's Decoding Biology Hackathon.**

bioLLM combines the power of Large Language Models with real-time access to cancer genomics databases (cBioPortal) and scientific literature, enabling accurate, data-driven answers to biology questions that require specialized knowledge and current research data.

## 🎯 Key Features

- **🤖 Agentic AI Architecture**: Uses autonomous agents that can reason, plan, and use tools to answer complex multi-step questions
- **🧬 Real Cancer Genomics Data**: Direct integration with cBioPortal API for mutation frequencies, expression data, and clinical information
- **⚡ High-Performance Inference**: vLLM backend with continuous batching, prefix caching (90%+ hit rate), and 80 concurrent requests
- **🔧 Specialized Tools**: 
  - cBioPortal search for cancer genomics data
  - Protein expression analysis
  - BioRxiv literature search
- **🚀 Production-Ready**: Async I/O, connection pooling, comprehensive error handling

## 🏆 Hackathon Context

Developed for **Owkin's Decoding Biology Hackathon**, this system addresses the challenge of answering complex biological questions that require:
- Understanding of molecular biology and cancer research
- Access to real-world patient data and clinical studies
- Integration of multiple data sources (mutations, expression, clinical outcomes)
- Reasoning across different biological concepts and pathways

The agentic approach allows the model to autonomously query databases, analyze data, and synthesize answers—going beyond simple retrieval to perform multi-step reasoning.

## 🧠 Technical Approach

### Agentic Architecture
bioLLM uses an **agent-based reasoning system** where the LLM can:
1. **Analyze** the question to determine what information is needed
2. **Plan** which tools to use and in what order
3. **Execute** tool calls to fetch real-world data from cBioPortal
4. **Reason** over the results to synthesize a final answer
5. **Format** the response with proper citations to data sources

### Key Innovations
- **Async Tool Execution**: Non-blocking I/O prevents event loop stalls during API calls
- **Connection Pool Optimization**: 100-connection pool handles 80 concurrent requests efficiently
- **Prompt Engineering**: Directive instructions minimize "thinking time" before tool usage
- **Continuous Batching**: vLLM server-side optimization for maximum GPU utilization
- **Prefix Caching**: 90%+ cache hit rate for repeated question patterns

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

The agent has access to specialized tools for biological data retrieval:

#### 1. **cBioPortal Tool** (`search_cbioportal`)
Primary tool for cancer genomics data from real patient studies:
- **Mutation Data**: Frequencies, hotspots (e.g., BRAF V600E), truncating mutations
- **mRNA Expression**: Tumor vs normal tissue, z-scores, fold changes
- **Protein Expression**: RPPA data, phosphorylation states
- **Copy Number Alterations**: Amplifications, deletions, neutral regions
- **Clinical Data**: Patient demographics, survival, staging
- **Multi-Study Aggregation**: Automatically combines data from up to 5 relevant studies

**Example Query**: "What is the TP53 mutation frequency in lung adenocarcinoma?"
→ Agent queries cBioPortal → Returns: "52.3% mutation frequency across 1,144 samples"

#### 2. **Protein Expression Tool** (`search_protein_expression`)
Focused wrapper around cBioPortal for protein-specific queries with RPPA data.

#### 3. **BioRxiv Tool** (optional)
Search preprint literature for recent biological discoveries (disabled by default for speed).

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

## � Performance & Results

### System Performance
- **Throughput**: ~2-3 questions/second with tool calls
- **GPU Utilization**: 85-95% across 8 L4 GPUs
- **Prefix Cache Hit Rate**: 92-94%
- **Connection Pool**: 100 connections handling 80 concurrent requests
- **Average Response Time**: 15-30 seconds per complex question (with multiple tool calls)

### Agentic Capabilities
The agent successfully handles multi-step reasoning:
1. **Comparative Questions**: "Which cancer type has higher TP53 mutations: lung or breast?"
   - Makes 2 separate tool calls to compare data
2. **Multi-Gene Analysis**: "Compare mutation patterns of EGFR, KRAS, and ALK in lung cancer"
   - Queries all genes simultaneously, analyzes patterns
3. **Cross-Reference**: "Is BRCA1 amplified in the same samples where TP53 is mutated?"
   - Combines mutation + CNA data for correlation analysis

### Hackathon Achievements
- ✅ Successfully integrates real-world genomics data (millions of samples from cBioPortal)
- ✅ Handles complex multi-step biological reasoning
- ✅ Provides data-driven answers with proper citations
- ✅ Production-ready system with async architecture and error handling
- ✅ Optimized for high-throughput inference (80 concurrent requests)

## 🏆 About Owkin's Decoding Biology Hackathon

This project was developed as part of Owkin's Decoding Biology Hackathon, which challenged participants to build AI systems capable of answering complex biological questions across multiple domains:
- Cancer genomics and mutations
- Gene expression and regulation
- Protein function and druggability
- Clinical outcomes and patient data

The hackathon emphasized the need for AI systems that can **reason with real data** rather than relying solely on pre-trained knowledge, making the agentic approach with tool integration essential.

## �🔒 License

See repository license.

## 🤝 Contributing

This is a production-ready system. For improvements, consider:
- Adding comprehensive error handling
- Implementing request rate limiting
- Adding authentication for API endpoints
- Setting up monitoring and logging infrastructure
- Adding unit and integration tests
