import subprocess
import sys
import time
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agent.core.agent import Agent
from agent.config import config


def start_rag_server():
    if config.RAG_BACKEND != "local_faiss":
        print("[UI] RAG_BACKEND is not 'local_faiss', skipping local server startup.")
        return None

    print("[UI] Starting RAG server on http://0.0.0.0:8000 ...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "agent.rag.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=str(ROOT),
    )
    time.sleep(2)
    return proc


def main():
    load_dotenv()
    agent = Agent("CognitiveAgent")
    rag_proc = start_rag_server()

    def run_agent_stream(user_query: str):
        if not user_query.strip():
            yield "Please enter a task."
            return

        buffer = []
        done = False

        def log_fn(msg: str):
            buffer.append(str(msg))

        # Run agent in a background thread
        def worker():
            nonlocal done
            try:
                agent.run(user_query, log_fn=log_fn)
            finally:
                done = True

        thread = threading.Thread(target=worker)
        thread.start()

        # Stream updates while the thread is running
        last_len = 0
        while not done or len(buffer) != last_len:
            if buffer:
                text = "\n".join(buffer)
                yield text
                last_len = len(buffer)
            time.sleep(0.3)  # update frequency

        # Final state (in case anything was appended at the end)
        if buffer:
            yield "\n".join(buffer)

    with gr.Blocks() as demo:
        gr.Markdown("# Cognitive Agent UI")
        gr.Markdown(
            "Enter a task. The agent will inspect hardware and a target project, "
            "then suggest memory and performance optimizations."
        )

        inp = gr.Textbox(
            label="Task",
            lines=4,
            value=(
                "Given this machine's hardware and a target project (local path or GitHub URL), "
                "analyze it and suggest concrete changes to improve memory usage and inference performance."
            ),
        )
        out = gr.Markdown(label="Agent Output")
        run_btn = gr.Button("Run Agent")

        run_btn.click(
            fn=run_agent_stream,
            inputs=inp,
            outputs=out,
        )

    try:
        demo.launch()
    finally:
        if rag_proc is not None:
            print("[UI] Stopping RAG server...")
            rag_proc.terminate()


if __name__ == "__main__":
    main()
