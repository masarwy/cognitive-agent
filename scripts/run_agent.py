from dotenv import load_dotenv
load_dotenv()

from agent.core.agent import Agent


def main():

    agent = Agent("CognitiveAgent")

    # agent.run("Research TensorRT-LLM performance optimizations")
    agent.run("Considering this machine hardware, What do you suggest to do in order to reduce model memory usage in the following folder '/home/mohammad-dr/PycharmProjects/devscribe'?")

if __name__ == "__main__":
    main()