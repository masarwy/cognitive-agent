# In agent/tools/code_analyze_tool.py

import requests
from agent.tools.llm_tool import LLMTool
from agent.config import config


class CodeAnalyzeTool(LLMTool):
    """Analyze code for memory usage patterns and optimization opportunities"""

    def __init__(self):
        system_prompt = (
            "You are an expert code analyzer specializing in memory optimization. "
            "Analyze code for memory usage patterns, identify issues, and suggest "
            "specific optimizations. Be precise and reference actual code patterns. "
            "Return ONLY the analysis without thinking process or meta-commentary."
        )
        super().__init__("code_analyze", system_prompt)
        self.rag_server_url = config.RAG_SERVER_URL  # e.g., http://localhost:8000

    def execute(self, step_description: str) -> str:
        """
        Retrieve relevant code from RAG service and analyze for memory patterns.

        Args:
            step_description: Description of what to analyze

        Returns:
            Structured analysis with specific findings
        """
        print(f"[{self.name}] Analyzing code...")

        # Extract search query from step description
        query = self._extract_query(step_description)

        # Retrieve relevant code from RAG service
        print(f"[{self.name}] Retrieving code for: {query}")
        results = self._query_rag(query, top_k=10)

        if not results:
            return "No code found in the ingested documents. Please run the ingest tool first."

        # Format retrieved code
        code_context = self._format_retrieved_code(results)

        # Build analysis prompt
        analysis_prompt = self._build_analysis_prompt(step_description, code_context)

        # Call parent's execute to get LLM response
        response = super().execute(analysis_prompt)

        return response  # Already cleaned by LLMTool base class

    def _query_rag(self, query: str, top_k: int = 5) -> list:
        """Query the RAG server for relevant code"""
        try:
            response = requests.post(
                f"{self.rag_server_url}/query",
                json={"query": query, "top_k": top_k},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.exceptions.RequestException as e:
            print(f"[{self.name}] RAG query failed: {e}")
            return []

    def _extract_query(self, step_description: str) -> str:
        """Extract search terms focusing on memory-related patterns"""
        memory_keywords = [
            "memory", "model", "tensor", "batch", "config",
            "cuda", "gpu", "load", "cache", "buffer", "allocation"
        ]

        # Find relevant keywords in description
        desc_lower = step_description.lower()
        query_terms = [kw for kw in memory_keywords if kw in desc_lower]

        # Default query if nothing specific found
        if not query_terms:
            query_terms = ["model", "config", "memory", "batch"]

        return " ".join(query_terms[:5])

    def _format_retrieved_code(self, results: list) -> str:
        """Format top retrieved code snippets"""
        if not results:
            return "No code snippets retrieved."

        formatted = []

        for i, result in enumerate(results[:5], 1):  # Top 5 results
            score = result.get('score', 0)
            text = result.get('text', '')

            formatted.append(
                f"=== Code Snippet {i} (Relevance: {score:.3f}) ===\n"
                f"{text}\n"
            )

        return "\n".join(formatted)

    def _build_analysis_prompt(self, task_description: str, code_context: str) -> str:
        """Build the analysis prompt for the LLM"""
        return f"""Task: {task_description}

Analyze the following code snippets for memory usage patterns and optimization opportunities.

{code_context}

Provide a structured analysis covering:

## 1. Memory Usage Patterns Found
- Large data structures or models
- Tensor allocations and shapes
- Batch size configurations
- Cache or buffer usage

## 2. Specific Issues Identified
- Memory-intensive operations with code references
- Potential memory leaks or retention issues
- Inefficient memory usage patterns
- Missing memory optimizations

## 3. Optimization Opportunities
- Specific recommendations with line/code references
- Expected memory savings (approximate if possible)
- Implementation complexity and trade-offs

Be specific and actionable. Reference actual code patterns you see."""
