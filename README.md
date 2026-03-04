# Cognitive Agent 🤖

A production-ready AI agent that analyzes codebases for memory optimization opportunities by combining hardware
profiling, code analysis, and contextualized reasoning.

## Features

- **Hardware-Aware Analysis**: Detects CPU, RAM, GPU (CUDA) specifications
- **GitHub Integration**: Clone and analyze any public repository
- **Smart Planning**: LLM-powered task decomposition
- **Framework Detection**: Automatically identifies ML/AI frameworks and libraries
- **Code Analysis**: Semantic search for memory and performance patterns with line numbers
- **Multi-Focus Analysis**: Supports memory optimization, inference latency, and general performance
- **Quantified Recommendations**: Specific optimizations with complexity estimates
- **Optional Web UI**: Simple Gradio-based interface for interacting with the agent

## What Can It Analyze?

The agent provides optimization recommendations for:

- **Memory Usage**: Detect memory leaks, large allocations, inefficient data structures
- **Inference Performance**: Identify latency bottlenecks, batch processing issues
- **ML/AI Systems**: PyTorch, TensorFlow, JAX, ONNX Runtime, TensorRT, OpenVINO
- **Data Processing**: pandas, numpy, Polars, Dask pipelines
- **Web Services**: FastAPI, Flask, Django applications
- **Any Python Codebase**: General performance and optimization opportunities

## Quick Start

### 1. Installation

```bash
git clone https://github.com/masarwy/cognitive-agent.git
cd cognitive-agent
pip install -r requirements.txt
```

Create a `.env` file in the root directory with the following settings:
```bash
# LLM Configuration
APP_LLM_SERVERURL=https://integrate.api.nvidia.com
APP_LLM_MODELNAME=nvidia/llama-3.3-nemotron-super-49b-v1.5
NVIDIA_API_KEY=your_nvidia_api_key_here

# RAG Backend Configuration
RAG_BACKEND=local_faiss              # Options: local_faiss, nvidia_rag
RAG_SERVER_URL=http://localhost:8000

# Embedding Configuration
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DEVICE=cuda                 # Options: cuda, cpu
```

### 2. Configuration Options

**LLM Settings:**

- `APP_LLM_SERVERURL`: NVIDIA API endpoint
- `APP_LLM_MODELNAME`: Model to use (default: Nemotron 49B)
- `NVIDIA_API_KEY`: Your NVIDIA API key ([Get one here](https://build.nvidia.com))

**RAG Backend:**

- `local_faiss`: Use local FAISS vector store (requires RAG service running)
- `nvidia_rag`: Use NVIDIA RAG service (no local server needed)

**Embeddings:**

- `EMBEDDING_MODEL`: Model for semantic search
- `EMBEDDING_DEVICE`: `cuda` for GPU, `cpu` for CPU-only

### 3. RAG Service Setup

**If using `local_faiss`**, start the RAG service in Terminal 1:

```bash
uvicorn agent.rag.server:app --host 0.0.0.0 --port 8000
```

**If using `nvidia_rag`**, skip the RAG service step - it's handled automatically. (not tested yet)

**Terminal 2** - Run agent:

```python
from dotenv import load_dotenv

load_dotenv()

from agent.core.agent import Agent

agent = Agent("CognitiveAgent")
agent.run("Considering this machine hardware, what do you suggest to reduce "
          "model memory usage in 'https://github.com/user/repo'?")
```

```bash
python -m scripts.run_agent
```

### 4. Web UI (optional, Gradio)

You can run an interactive web UI using Gradio:

```bash
python -m scripts.run_gradio_ui
```

This will:

- Start the local RAG server (if RAG_BACKEND=local_faiss)

- Launch a browser UI where you can enter tasks

- Stream the agent’s step-by-step output into the page

Navigate to http://localhost:7860/ to view the UI.

## How It Works

1. **Planning**: LLM creates execution plan from your query
2. **Execution**: Tools run sequentially with context passing
3. **Hardware Detection**: Profiles CPU, RAM, GPU, CUDA capabilities, Tensor Cores
4. **Framework Detection**: Identifies ML/AI frameworks from code and dependencies
5. **Code Analysis**: Semantic search finds memory patterns and performance bottlenecks
6. **Reasoning**: Generates hardware-aware, framework-specific recommendations

## Example Output

```
Framework Detection
Detected frameworks: PyTorch 2.10.0, Sentence Transformers 5.2.2, ChromaDB 1.4.1

Executive Summary
Hardware: 15GB RAM, RTX 4050 (6GB VRAM, Compute 8.9, Tensor Cores available)
Opportunity: Reduce memory by 30-50% through model optimization and streaming processing

Key Recommendations

1. Model FP16 Conversion (High Impact, Medium Complexity)
   - Location: chroma_test.py:12-13
   - Savings: 1.5GB VRAM (40-60% reduction)
   - Code: model.half().to("cuda")

2. Streaming File Processing (High Impact, High Complexity)
   - Location: rag.py:63-100
   - Savings: O(1) memory usage
   - Implementation: Replace full file reads with generators

3. Batched Collection Operations (Medium Impact, Low Complexity)
   - Location: chroma_test.py:30-33
   - Savings: 40% faster ingestion
   - Complexity: Low

4. Context Compression (Medium Impact, Low Complexity)
   - Location: llm.py:25-30
   - Savings: 200MB per context
   - Trade-off: Requires decompression
```

## Use Cases

### Memory Optimization

```python
from agent.core.agent import Agent

agent = Agent("MemoryOptimizationAgent")
agent.run("Reduce memory usage in https://github.com/user/ml-project")
```

### Inference Performance

```python
from agent.core.agent import Agent

agent = Agent("InferencePerformanceAgent")
agent.run("Optimize inference latency in https://github.com/user/api-service")
```

### Local Analysis

```python
from agent.core.agent import Agent

agent = Agent("LocalAnalysisAgent")
agent.run("Analyze memory usage in '/path/to/local/project'")
```

## Available Tools

| Tool               | Purpose                                   |
|--------------------|-------------------------------------------|
| `github_clone`     | Clone repositories                        |
| `ingest`           | Index code files for semantic search      |
| `code_analyze`     | Detect frameworks and find patterns       |
| `hardware_analyze` | Profile CPU, RAM, GPU, CUDA, Tensor Cores |
| `reason`           | Generate hardware-aware recommendations   |
| `retrieve`         | Semantic code search                      |

## Troubleshooting

**RAG service error**: Ensure service is running on port 8000

**CUDA out of memory**: Set `CUDA_VISIBLE_DEVICES=""` for CPU mode

**No files found**: Verify folder path and supported file types (`.py`, `.txt`, `.json`, `.yaml`, `.yml`)

## License

Apache License 2.0 - see [LICENSE](LICENSE)

## Contact

- Issues: [GitHub Issues](https://github.com/masarwy/cognitive-agent/issues)
- Author: Mohammad Masarwy

---

⭐ Star this repo if you find it useful!
