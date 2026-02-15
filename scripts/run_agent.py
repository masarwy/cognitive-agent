from dotenv import load_dotenv
load_dotenv()

from agent.core.agent import Agent


def main():

    agent = Agent("CognitiveAgent")

    agent.run("Research TensorRT-LLM performance optimizations")


if __name__ == "__main__":
    main()