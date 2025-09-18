#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from crew import Videogeneration
from langchain_openai import ChatOpenAI
import os

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

####################################################################################################
##################################### Load Model Parameters ########################################
####################################################################################################
# importing necessary functions from dotenv library
from dotenv import load_dotenv 
# loading variables from .env file
load_dotenv() 

# accessing and printing value
model = os.getenv('MODEL')
api_key = os.getenv('API_KEY')

os.environ["OPEN_API_KEY"] = "sk-proj-1111"

# llm = LLM(model=model, api_key=api_key, temperature=0)
llm = ChatOpenAI(
    model = "llama3.1",
    base_url = "http://localhost:11434/v1"
)

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }
    
    try:
        Videogeneration(llm=llm).crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        Videogeneration(llm=llm).crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Videogeneration(llm=llm).crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        Videogeneration(llm=llm).crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":

    # Warning control
    import warnings
    warnings.filterwarnings('ignore')

    run()