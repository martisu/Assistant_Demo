from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
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
                "If it's not in the scope of repair or renovation then ask the quesiton 'No he entendido tu pregunta, pregunta de nuevo '"
                "DO NOT ANSWER something inrelated to construction of houses"
            ),
            llm=self.llm
        )

    def repair_agent(self):
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

    def renovation_agent(self):
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
    def materials_agent(self):
        return Agent(
            role='Materials Expert',
            goal='Provide a detailed list of materials used for the job.',
            tools=[file_read_tool, scrape_tool, search_tool],
            verbose=True,
            backstory=(
                "You are an experienced expert in construction. "
                "Your role is to provide detailed advice on materials used for various projects. "
                "Create a list using markdown that includes the materials and their alternatives. "
                "Always respond in Spanish unless otherwise indicated. "
                "Use a maximum of four sentences to keep the response concise."
            ),
            llm=self.llm
        )

    
    def tools_agent(self):
        return Agent(
            role='Tools Expert',
            goal='Provide a detailed list of tools used for the job.',
            tools=[file_read_tool, scrape_tool, search_tool],
            verbose=True,
            backstory=(
                "You are an experienced expert in construction. "
                "Your role is to provide detailed advice on tools required for various tasks. "
                "Create a list using markdown that includes the tools and their alternatives. "
                "Always respond in Spanish unless otherwise indicated. "
                "Use a maximum of four sentences to keep the response concise."
            ),
            llm=self.llm
        )

    
    def cost_agent(self):
        return Agent(
            role='Cost Determinator',
            goal='Based on the list of materials, provide a table with the costs.',
            tools=[file_read_tool, scrape_tool, search_tool],
            verbose=True,
            backstory=(
                "You are a cost expert in construction. "
                "Your role is to provide detailed cost estimations for materials used. "
                "Create a list using markdown that includes the costs of each material and their alternatives. "
                "Always respond in Spanish unless otherwise indicated. "
                "Use a maximum of four sentences to keep the response concise."
            ),
            llm=self.llm
        )
    def step_by_step_agent(self):
        return Agent(
            role='Step-by-Step Guide',
            goal='Provide detailed step-by-step instructions for any task.',
            tools=[file_read_tool, scrape_tool, search_tool],
            verbose=True,
            backstory=(
                "You are an expert guide. Your role is to break down complex tasks into clear, manageable steps."
                "Always ensure the instructions are simple and precise, adjusting based on the user's feedback."
                "Provide explanations for each step, but keep them concise."
                "Respond in Spanish unless otherwise indicated."
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
        agent = self.repair_agent() if project_type == 'repair' else self.renovation_agent()
        return Task(
            description=f"Provide detailed guidance for this {project_type} project: {question}",
            agent=agent,
            expected_output=(
                "A comprehensive guide for the home improvement project, including step-by-step instructions, "
                "required materials, estimated time and cost, and any safety precautions."
            )
        )
    def list_materials_task(self, project_description):
        return Task(
            description=f"List the materials required for the following project: {project_description}. "
                        f"It should only contain the MATERIALS, DO NOT add the TOOLS"
                        f"Include alternatives where applicable.",
            agent=self.materials_agent(),
            expected_output=(
                "A markdown list of materials and their alternatives, e.g.,:\n\n"
                "- **Material 1**: Description\n  - Alternative: Option 1\n  - Alternative: Option 2\n"
            ),
            # human_input=True
        )

    def list_tools_task(self, project_description):
        return Task(
            description=f"List the tools required for the following project: {project_description}. "
                        f"It should only contain the TOOLS, DO NOT add the MATERIALS",
            agent=self.tools_agent(),
            expected_output=(
                "A markdown list of tools and their alternatives, e.g.,:\n\n"
                "- **Tool 1**: Description\n  - Alternative: Option 1\n  - Alternative: Option 2\n"
            ),
            # human_input=True
        )

    def cost_estimation_task(self, materials_list):
        return Task(
            description=f"Provide a detailed cost estimation for the following materials: {materials_list}. "
                        f"Include costs for alternatives where applicable.",
            agent=self.cost_agent(),
            expected_output=(
                "A markdown table listing materials and their costs, including alternatives, e.g.,:\n\n"
                "| Material        | Cost  | Alternatives               |\n"
                "|----------------|-------|----------------------------|\n"
                "| Material 1     | $10   | Alternative 1 ($8), Alt 2 ($12) |\n"
            ),
            # human_input=True
        )
    def step_by_step_task(self, repair_or_renovation_process):
        return Task(
            description=f"Provide detailed step-by-step instructions for the following repair or renovation process: {repair_or_renovation_process}. "
                        f"Ensure that the steps are easy to follow and comprehensive, covering all necessary tools, materials, and safety precautions.",
            agent=self.step_by_step_agent(),
            expected_output=(
                "A list of detailed steps for the repair or renovation process. For example:\n\n"
                "1. Identify the scope of the repair or renovation.\n"
                "2. Gather all necessary tools and materials.\n"
                "3. Prepare the work area to ensure safety and efficiency.\n"
                "4. Step-by-step breakdown of the actual work (e.g., removing old materials, installing new ones).\n"
                "5. Final touches and clean-up instructions.\n"
            ),
            # human_input=True  # Uncomment if you want to enable human input
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