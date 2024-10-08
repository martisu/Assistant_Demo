from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import yaml

class CrewAIChatbot:
    def __init__(self, credentials_path):
        with open(credentials_path, "r") as stream:
            credentials = yaml.safe_load(stream)
        
        self.openai_api_key = credentials["OPENAI_API_KEY"]
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=self.openai_api_key)

    def create_agent(self):
        return Agent(
            role='Home Improvement Assistant',
            goal='Provide helpful advice on home improvement projects',
            backstory="You are an AI assistant specialized in home improvement and DIY projects.",
            verbose=True,
            llm=self.llm
        )

    def create_task(self, question):
        agent = self.create_agent()
        return Task(
            description=f"Answer the following question about home improvement: {question}",
            agent=agent,
            expected_output="A detailed and helpful response to the user's home improvement question."
        )

    def get_response(self, question):
        try:
            task = self.create_task(question)
            crew = Crew(
                agents=[task.agent],
                tasks=[task],
                verbose=2
            )
            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return f"I'm sorry, but I encountered an error while processing your request: {str(e)}"