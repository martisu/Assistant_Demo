from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import ScrapeWebsiteTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools import Tool
import yaml
import os

class CrewAIChatbot:
    def __init__(self, credentials_path):
        self.credentials = self.load_credentials(credentials_path)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=self.credentials["OPENAI_API_KEY"])
        self.search_tool = DuckDuckGoSearchRun()
        self.scrape_tools = self.load_scrape_tools()
        self.pdf_tools = self.load_pdf_tools()

    def load_credentials(self, path):
        with open(path, "r") as stream:
            return yaml.safe_load(stream)

    def load_scrape_tools(self):
        with open("data/sites/websites.yaml", "r") as file:
            websites = yaml.safe_load(file)

        scrape_tools = []
        for resource in websites['resources']:
            scrape_tools.append(ScrapeWebsiteTool(
                website_url=resource['url'],
                website_name=resource['name'],
                website_description=resource['description']
            ))
        return scrape_tools

    def load_pdf_tools(self):
        pdf_tools = []
        pdf_dir = "data/"
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, filename)
                loader = PyPDFLoader(pdf_path)
                pdf_tool = Tool(
                    name=f"PDF_Reader_{filename}",
                    func=loader.load,
                    description=f"Use this tool to read and extract information from the PDF file {filename}"
                )
                pdf_tools.append(pdf_tool)
        return pdf_tools
    ##------------------------------------AGENTS------------------------------------
    def planificator(self):
        return Agent(
            role='Project Classifier',
            goal='Classify whether a home improvement project is a repair or a renovation.',
            tools=[self.search_tool],
            verbose=True,
            backstory=(
                "You are an expert in classifying home improvement projects. "
                "Your task is to determine if a project is a **repair** (fixing or restoring something damaged) "
                "or a **renovation** (improving or modernizing an existing feature) based on its scope and complexity. "
                "Always respond in Spanish unless otherwise indicated. "
                "If you don't know the answer, just say you don't know, without making one up. "
                "Use a maximum of one line per response to keep it concise. "
                "Always end your response with 'another question?' in the specified language."
            ),
            llm=self.llm
        )

    def repair_expert(self):
        return Agent(
            role='Repair Expert',
            goal='Provide detailed guidance on home repair projects.',
            tools=[self.search_tool] + self.scrape_tools + self.pdf_tools,
            verbose=True,
            backstory=(
                "You are an experienced expert in home repairs. "
                "Your role is to provide clear and practical guidance on repair projects, "
                "focusing on fixes and maintenance tasks that do not require major structural changes. "
                "Always respond in Spanish unless otherwise indicated. "
                "Use a maximum of four sentences to keep the response concise. "
                "Always end your response with 'another question?' in the specified language."
            ),
            llm=self.llm
        )


    def renovation_expert(self):
        return Agent(
            role='Renovation Expert',
            goal='Provide detailed guidance on home renovation projects.',
            tools=[self.search_tool] + self.pdf_tools,
            verbose=True,
            backstory=(
                "You are an experienced expert in home renovations. "
                "Your role is to provide in-depth advice on renovation projects, "
                "particularly those involving major structural changes or additions to the home. "
                "Always respond in Spanish unless otherwise indicated. "
                "Use a maximum of four sentences to keep the response concise. "
                "Always end your response with 'another question?' in the specified language."
            ),
            llm=self.llm
        )

    
    def materials_agent(self):
        return Agent(
            role='Materials Expert',
            goal='Provide a detailed list of materials used for the job.',
            tools=[self.search_tool] + self.scrape_tools + self.pdf_tools,
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

    
    def herramientas_agent(self):
        return Agent(
            role='Tools Expert',
            goal='Provide a detailed list of tools used for the job.',
            tools=[self.search_tool] + self.scrape_tools + self.pdf_tools,
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
            tools=[self.search_tool] + self.scrape_tools + self.pdf_tools,
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


##------------------------------------TASKS------------------------------------

    def classify_project_task(self, question):
        return Task(
            description=f"Classify the following home improvement project: {question}. "
                        f"You can use synonym pages to find keywords similar to renovation or repair.",
            agent=self.planificator(),
            expected_output="Project classified as 'repair' or 'renovation'."
        )

    def provide_guidance_task(self, question, project_type):
        agent = self.repair_expert() if project_type == 'repair' else self.renovation_expert()
        return Task(
            description=f"Provide detailed guidance for this {project_type} project: {question}.",
            agent=agent,
            expected_output=(
                "A complete guide for the home improvement project, including step-by-step instructions, "
                "required materials, estimated time and cost, and any safety precautions."
            ),
            # human_input=True
        )

    def list_materials_task(self, project_description):
        return Task(
            description=f"List the materials required for the following project: {project_description}. "
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
                        f"Include alternatives where applicable.",
            agent=self.herramientas_agent(),
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

    ##------------------------------------CREATE CREW------------------------------------

    def get_response(self, question):
        try:
            # Step 1: Classify the project
            classification_task = self.classify_project_task(question)
            crew = Crew(
                agents=[classification_task.agent],
                tasks=[classification_task],
                verbose=True
            )
            result = crew.kickoff()
            project_type = 'repair' if 'repair' in result.lower() else 'renovation'

            # Step 2: Provide guidance
            guidance_task = self.provide_guidance_task(question, project_type)
            
            # Step 3: List materials
            materials_task = self.list_materials_task(question)
            
            # Step 4: List tools
            tools_task = self.list_tools_task(question)
            
            # Step 5: Cost estimation
            cost_task = self.cost_estimation_task("Materials from the previous task")

            # Create the main crew
            home_improvement_crew = Crew(
                agents=[guidance_task.agent, materials_task.agent, tools_task.agent, cost_task.agent],
                tasks=[guidance_task, materials_task, tools_task, cost_task],
                verbose=2
            )
            

            result = home_improvement_crew.kickoff()

            return result
        except Exception as e:
            return f"Perdon, encontre un error en encontrar una solucion : {str(e)}"
