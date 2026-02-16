import requests
from pathlib import Path
from agent.tools.base import Tool
from agent.config import config

class IngestTool(Tool):
    """
    Tool to ingest local files into the RAG server so RetrieveTool can query them.
    Supports code repos, text files, and optionally other document types.
    """

    def __init__(self):
        super().__init__("ingest")
        self.server_url = config.RAG_SERVER_URL  # same server used by RetrieveTool

    def execute(self, source_path: str, file_types=None) -> str:
        """
        :param source_path: local directory or file path to ingest
        :param file_types: list of extensions to include, e.g., ['.py', '.json']
        """
        file_types = file_types or ['.py', '.txt', '.json', '.yaml', '.yml']

        print(f"[IngestTool] Scanning {source_path} for files: {file_types}")

        # Collect files
        all_files = []
        for path in Path(source_path).rglob("*"):
            if path.is_file() and path.suffix in file_types:
                all_files.append(path)

        if not all_files:
            return f"No files found in {source_path} matching types {file_types}"

        # Read files and prepare for ingestion
        docs_to_ingest = []
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        docs_to_ingest.append({
                            "path": str(file_path),
                            "text": content
                        })
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

        # Send documents to RAG server
        ingest_url = f"{self.server_url}/ingest"
        print(f"[IngestTool] Sending {len(docs_to_ingest)} documents to RAG server...")

        payload = {"documents": docs_to_ingest}

        response = requests.post(ingest_url, json=payload)
        response.raise_for_status()

        return f"Successfully ingested {len(docs_to_ingest)} documents into RAG server."