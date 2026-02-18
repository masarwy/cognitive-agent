# Cognitive Agent 🤖

A production-ready AI agent that analyzes codebases for memory optimization opportunities by combining hardware
profiling, code analysis, and contextualized reasoning.

## Features

- **Hardware-Aware Analysis**: Detects CPU, RAM, GPU (CUDA) specifications
- **GitHub Integration**: Clone and analyze any public repository
- **Smart Planning**: LLM-powered task decomposition
- **Code Analysis**: Semantic search for memory patterns with line numbers
- **Quantified Recommendations**: Specific optimizations with complexity estimates

## Quick Start

### 1. Installation

```bash
git clone https://github.com/masarwy/cognitive-agent.git
cd cognitive-agent
pip install -r requirements.txt
```

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

### Configuration Options

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

### RAG Service Setup

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

## How It Works

1. **Planning**: LLM creates execution plan from your query
2. **Execution**: Tools run sequentially with context passing
3. **Analysis**: Code patterns + hardware specs analyzed
4. **Recommendations**: Quantified, prioritized optimizations

## Example Output

```
Executive Summary: 
System has 15GB RAM, 6GB GPU. Reduce memory by optimizing model loading 
and string operations.

Key Recommendations:
1. Truncate Context (Low, 50-90% savings) - llm.py:20-25
2. Limit History (Low) - cli.py:150  
3. Smaller Model (Medium, 50% savings) - rag.py:43
```

## Available Tools

| Tool               | Purpose                  |
|--------------------|--------------------------|
| `github_clone`     | Clone repositories       |
| `ingest`           | Index code files         |
| `code_analyze`     | Find memory patterns     |
| `hardware_analyze` | Profile system           |
| `reason`           | Generate recommendations |

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
