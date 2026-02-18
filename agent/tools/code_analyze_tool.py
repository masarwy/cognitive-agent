from agent.tools.llm_tool import LLMTool
import requests
import os


class CodeAnalyzeTool(LLMTool):
    """
    Analyze code for performance and memory optimization opportunities.
    Detects frameworks and provides contextualized recommendations.
    """

    def __init__(self):
        system_prompt = """You are an expert code analyzer specializing in software optimization.

Your expertise covers:
- Memory usage patterns and optimization
- Performance optimization and latency reduction
- System bottlenecks and inefficiencies
- Framework-specific best practices

## Analysis Process:

### 1. Framework/Library Detection (REQUIRED FIRST)
Identify key frameworks and libraries used in the codebase:
- ML/AI frameworks (PyTorch, TensorFlow, JAX, scikit-learn)
- Inference engines (ONNX Runtime, TensorRT, OpenVINO)
- Data processing (pandas, numpy, Polars, Dask)
- Web frameworks (FastAPI, Flask, Django)
- Database libraries (SQLAlchemy, asyncpg, Redis)
- Other significant dependencies

List detected frameworks/libraries with versions if available.

### 2. Performance Analysis
Identify bottlenecks in:
- Data loading and initialization overhead
- Input preprocessing operations (parsing, validation, transformation)
- Batch processing inefficiencies (batch size, parallelization)
- Core computation or business logic execution
- Output processing and serialization
- Sequential vs parallel execution opportunities
- I/O operations (file, network, database)

### 3. Memory Analysis
Find memory-intensive patterns:
- Large data structures or in-memory caches
- Object allocations and data types
- Batch/chunk size configurations
- Buffer usage and memory pools
- Memory leaks or retention issues
- Inefficient memory usage patterns (copies, redundant storage)

### 4. Optimization Opportunities
Provide framework-specific recommendations with:
- Code references (file paths and line numbers)
- Expected savings (percentage, memory size, or timing)
- Implementation complexity (Low/Medium/High)
- Trade-offs (performance vs accuracy/reliability, memory vs speed)

## Output Format:

Start with "## Framework Detection" section, then organize findings by:
1. Memory Usage Patterns Found
2. Performance Bottlenecks Identified
3. Optimization Opportunities (prioritized by impact)
"""
        super().__init__("code_analyze", system_prompt)
        self.rag_server = os.getenv("RAG_SERVER_URL", "http://localhost:8000")

    def execute(self, step_description: str) -> str:
        """
        Analyze code for optimization opportunities.

        Args:
            step_description: Task description, may include context from previous steps

        Returns:
            Detailed analysis with framework detection and optimization recommendations
        """
        print(f"[{self.name}] Analyzing code...")

        # Determine analysis focus from description
        focus = self._determine_analysis_focus(step_description)

        # Build search query based on focus
        query = self._build_search_query(focus)

        print(f"[{self.name}] Retrieving code for: {query}")

        # Retrieve relevant code contexts
        contexts = self._retrieve_contexts(query, k=7)

        if not contexts:
            return "No code found in the ingested data. Please ensure the code has been ingested first."

        print(f"[{self.name}] Processing...")

        # Build analysis prompt
        analysis_prompt = self._build_analysis_prompt(step_description, contexts)

        # Call LLM for analysis
        result = super().execute(analysis_prompt)

        return result

    def _determine_analysis_focus(self, description: str) -> str:
        """Determine what to focus the analysis on"""
        description_lower = description.lower()

        if 'memory' in description_lower:
            return 'memory'
        elif 'inference' in description_lower or 'latency' in description_lower:
            return 'inference'
        elif 'performance' in description_lower or 'speed' in description_lower:
            return 'performance'
        else:
            return 'general'

    def _build_search_query(self, focus: str) -> str:
        """Build search query based on analysis focus"""

        queries = {
            'memory': 'memory tensor cuda gpu allocation cache buffer',
            'inference': 'inference latency forward model.eval() throughput batch prediction',
            'performance': 'performance optimization bottleneck slow loop parallel',
            'general': 'optimization performance memory'
        }

        return queries.get(focus, queries['general'])

    def _retrieve_contexts(self, query: str, k: int = 7) -> list:
        """
        Retrieve relevant code contexts from RAG service.

        Args:
            query: Search query
            k: Number of results to retrieve

        Returns:
            List of context dictionaries with 'text' and 'path' keys
        """
        try:
            # Use /query endpoint instead of /retrieve
            response = requests.post(
                f"{self.rag_server}/query",
                json={"query": query, "top_k": k},  # Changed from "k" to "top_k"
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # Adapt response format from your server
            results = data.get("results", [])

            # Convert to expected format
            contexts = []
            for result in results:
                contexts.append({
                    'text': result.get('text', ''),
                    'path': result.get('metadata', {}).get('path', 'unknown')  # Adjust based on your metadata structure
                })

            return contexts

        except requests.exceptions.RequestException as e:
            print(f"[{self.name}] Error retrieving contexts: {e}")
            return []

        except requests.exceptions.RequestException as e:
            print(f"[{self.name}] Error retrieving contexts: {e}")
            return []

    def _build_analysis_prompt(self, task: str, contexts: list) -> str:
        """
        Build the analysis prompt with task and code contexts.

        Args:
            task: User's analysis task
            contexts: Retrieved code contexts

        Returns:
            Formatted prompt string
        """
        # Format contexts
        context_texts = []
        for i, ctx in enumerate(contexts):
            path = ctx.get('path', 'unknown')
            text = ctx.get('text', '')

            # Truncate very long contexts
            if len(text) > 2000:
                text = text[:2000] + "\n... [truncated]"

            context_texts.append(f"CONTEXT {i} (from {path}):\n{text}\n")

        context_block = "\n\n".join(context_texts)

        # Build full prompt
        prompt = f"""Task: {task}

Available Code Context:
{context_block}

Provide detailed analysis following your system instructions. Start with Framework Detection, then analyze memory patterns, performance bottlenecks, and optimization opportunities."""

        return prompt
