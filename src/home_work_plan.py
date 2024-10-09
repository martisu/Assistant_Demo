from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import yaml
from crewai_tools import FileReadTool, ScrapeWebsiteTool

search_tool = DuckDuckGoSearchRun()
scrape_tool = ScrapeWebsiteTool()
file_read_tool = FileReadTool(root_dir="data/")

class CrewAIChatbot:
    def __init__(self, credentials_path):
        with open(credentials_path, "r") as stream:
            credentials = yaml.safe_load(stream)
        
        self.openai_api_key = credentials["OPENAI_API_KEY"]
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=self.openai_api_key)

# AGENTS
    def planificator(self):
        return Agent(
            role='Project Classifier',
            goal='Determine whether a home improvement project is a repair or renovation.',
            tools=[search_tool],
            verbose=True,
            backstory=(
                "You are an expert in classifying home improvement projects. "
                "Your role is to determine whether a project is a repair or a renovation based on its scope and complexity."
                "Always use the Spanish language unless they tell you otherwise."
                "Use 4 sentences maximum but keep the answer as concise as possible."
                "Always say 'other question' in the specified language at the end of the answer."
            ),
            llm=self.llm
        )

    def repair_expert(self):
        return Agent(
            role='Repair Expert',
            goal='Provide detailed guidance on home repair projects.',
            tools=[file_read_tool, scrape_tool, search_tool],
            verbose=True,
            backstory=(
                "You are a seasoned home repair expert. You provide comprehensive guidance on repair projects, "
                "focusing on fixes and maintenance tasks that don't require significant structural changes."
                "Always use the Spanish language unless they tell you otherwise."
                "Use 4 sentences maximum but keep the answer as concise as possible."
                "Always say 'other question' in the specified language at the end of the answer."
            ),
            llm=self.llm
        )

    def renovation_expert(self):
        return Agent(
            role='Renovation Expert',
            goal='Provide detailed guidance on home renovation projects.',
            tools=[file_read_tool, scrape_tool, search_tool],
            verbose=True,
            backstory=(
                "You are an experienced home renovation expert. You offer in-depth advice on renovation projects, "
                "especially those involving significant structural changes or additions to the home."
                "Always use the Spanish language unless they tell you otherwise."
                "Use 4 sentences maximum but keep the answer as concise as possible."
                "Always say 'other question' in the specified language at the end of the answer."
            ),
            llm=self.llm
        )
# TASKS
    def classify_project_task(self, question):
        agent = self.planificator()
        return Task(
            description=f"Classify the following home improvement project: {question}",
            agent=agent,
            expected_output="Classification of the project as either 'repair' or 'renovation'."
        )

    def provide_guidance_task(self, question, project_type):
        agent = self.repair_expert() if project_type == 'repair' else self.renovation_expert()
        return Task(
            description=f"Provide detailed guidance for this {project_type} project: {question}",
            agent=agent,
            expected_output=(
                "A comprehensive guide for the home improvement project, including step-by-step instructions, "
                "required materials, estimated time and cost, and any safety precautions."
            )
        )
# CREATE CREW
    def get_response(self, question):
        try:
            # First, classify the project
            classification_task = self.classify_project_task(question)
            classification_crew = Crew(
                agents=[classification_task.agent],
                tasks=[classification_task],
                verbose=2
            )
            project_type = classification_crew.kickoff()

            # Then, provide guidance based on the classification
            guidance_task = self.provide_guidance_task(question, project_type)
            guidance_crew = Crew(
                agents=[guidance_task.agent],
                tasks=[guidance_task],
                verbose=2
            )
            result = guidance_crew.kickoff()

            return result
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return f"Lo siento he encontrado un error al buscar esta solucion: {str(e)}"