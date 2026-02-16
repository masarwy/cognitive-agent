import os
import re
import requests
from pathlib import Path
from typing import List, Optional

from agent.tools.base import Tool
from agent.config import config

EXCLUDED_DIRS = {
    '.venv', 'venv', 'env',           # Virtual environments
    'node_modules',                     # Node.js packages
    '__pycache__', '.pytest_cache',    # Python cache
    '.git', '.svn', '.hg',             # Version control
    'build', 'dist', '.egg-info',      # Build artifacts
    '.idea', '.vscode',                # IDE configs
    'site-packages',                    # Python packages
}

class IngestTool(Tool):
    """
    Tool to ingest local files into the RAG server so RetrieveTool can query them.
    Supports code repos, text files, and optionally other document types.
    """

    def __init__(self):
        super().__init__("ingest")
        self.server_url = config.RAG_SERVER_URL  # same server used by RetrieveTool

    def execute(self, step_description: str, file_types: Optional[List[str]]=None) -> str:
        """
        :param source_path: local directory or file path to ingest
        :param file_types: list of extensions to include, e.g., ['.py', '.json']
        """
        file_types = file_types or ['.py', '.txt', '.json', '.yaml', '.yml']

        # Look for paths in quotes or after "folder"/"directory"
        path_match = re.search(r"['\"]([^'\"]+)['\"]", step_description)
        if path_match:
            folder_path = path_match.group(1)
        else:
            # Fallback: try to find path-like strings
            path_match = re.search(r"(/[\w\-/]+)", step_description)
            if path_match:
                folder_path = path_match.group(1)
            else:
                return f"Error: Could not extract folder path from: {step_description}"

        # Now use folder_path for scanning
        print(f"[IngestTool] Scanning {folder_path} for files: {file_types}")

        # Collect files
        all_files = self._scan_directory(folder_path, file_types)

        if not all_files:
            return f"No files found in {folder_path} matching types {file_types}"

        # Read files and prepare for ingestion
        documents = self._prepare_documents(all_files)

        # Batch the documents
        BATCH_SIZE = 100  # Adjust based on your server's capacity
        total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"[IngestTool] Sending {len(documents)} documents in {total_batches} batches...")

        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            print(f"[IngestTool] Batch {batch_num}/{total_batches}...")

            response = requests.post(
                f"{self.server_url}/ingest",
                json={"documents": batch}
            )
            response.raise_for_status()

        return f"Successfully ingested {len(documents)} documents from {folder_path}"

    def _scan_directory(self, folder_path: str, supported_extensions: Optional[List[str]]) -> List[str]:
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            # Filter out excluded directories IN-PLACE
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for filename in filenames:
                if any(filename.endswith(ext) for ext in supported_extensions):
                    files.append(os.path.join(root, filename))
        return files

    def _prepare_documents(self, all_files: List[str]) -> List[str]:
        docs_to_ingest = []
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        # Format: "filepath:\nContent"
                        doc_text = f"File: {file_path}\n\n{content}"
                        docs_to_ingest.append(doc_text)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
        return docs_to_ingest