from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from typing import List
import os
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Videogeneration():
    """Videogeneration crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self, llm: LLM):
        self.llm = llm

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def idea_creator(self) -> Agent:
        return Agent(
            config=self.agents_config['idea_creator'], # type: ignore[index]
            verbose=True
        )

    @agent
    def scene_creator(self) -> Agent:
        return Agent(
            config=self.agents_config['scene_creator'], # type: ignore[index]
            verbose=True
        )

    @agent
    def prompt_creator(self) -> Agent:
        return Agent(
            config=self.agents_config['prompt_creator'], # type: ignore[index]
            verbose=True
        )
    
    @agent
    def Caption_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['Caption_writer'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def new_idea_task(self) -> Task:
        return Task(
            config=self.tasks_config['new_idea_task'], # type: ignore[index]
        )

    @task
    def scene_creation(self) -> Task:
        return Task(
            config=self.tasks_config['scene_creation'], # type: ignore[index]
            output_file='report.md'
        )

    @task
    def prompt_creation(self) -> Task:
        return Task(
            config=self.tasks_config['prompt_creation'], # type: ignore[index]
            output_file='report.md'
        )

    @task
    def caption_task(self) -> Task:
        return Task(
            config=self.tasks_config['caption_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Videogeneration crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
