import os
import subprocess
from agent.tools.base import Tool


class GitHubCloneTool(Tool):
    """Clone a GitHub repository to local filesystem"""

    def __init__(self):
        super().__init__("github_clone")

    def execute(self, step_description: str) -> str:
        """
        Clone a GitHub repo from URL.
        Expects description like: "Clone repo https://github.com/user/repo"
        """
        print(f"[{self.name}] Cloning repository...")

        url = self._extract_github_url(step_description)

        if not url:
            return "Error: No valid GitHub URL found in step description"

        repo_name = url.rstrip('/').split('/')[-1].replace('.git', '')
        clone_path = os.path.join('/tmp', repo_name)

        if os.path.exists(clone_path):
            print(f"[{self.name}] Removing existing clone at {clone_path}")
            subprocess.run(['rm', '-rf', clone_path])

        try:
            result = subprocess.run(
                ['git', 'clone', url, clone_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return f"Error cloning repository: {result.stderr}"

            if not os.path.exists(clone_path):
                return "Error: Repository clone failed"

            file_count = sum(1 for _ in self._count_files(clone_path))

            return (
                f"Successfully cloned {repo_name} to {clone_path} ({file_count} files)\n\n"
                f"IMPORTANT: Repository is located at: {clone_path}\n"
                f"For subsequent steps, use the path: {clone_path}"
            )

        except subprocess.TimeoutExpired:
            return "Error: Clone operation timed out (repository may be too large)"
        except Exception as e:
            return f"Error: {str(e)}"

    def _extract_github_url(self, text: str) -> str:
        """Extract GitHub URL from text"""
        import re

        pattern = r'https?://github\.com/[\w-]+/[\w.-]+'
        match = re.search(pattern, text)

        return match.group(0) if match else ""

    def _count_files(self, directory: str):
        """Generator to count files"""
        for root, dirs, files in os.walk(directory):
            if '.git' in root:
                continue
            for file in files:
                yield os.path.join(root, file)