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
            goal='Determine whether a home improvement project is a repair or a renovation.',
            tools=[self.search_tool],
            verbose=True,
            backstory=(
                "You are an expert in classifying home improvement projects. "
                "Your role is to determine if a project is a repair or a renovation based on its scope and complexity."
                "Always respond in Spanish unless otherwise indicated."
                "Use a maximum of 4 sentences, but keep the response as concise as possible."
                "Always end with 'otra pregunta' (another question) in the specified language."
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
                "You are an experienced expert in home repairs. You provide comprehensive guidance on repair projects, "
                "focusing on fixes and maintenance tasks that do not require significant structural changes."
                "Always respond in Spanish unless otherwise indicated."
                "Use a maximum of 4 sentences, but keep the response as concise as possible."
                "Always end with 'otra pregunta' (another question) in the specified language."
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
                "You are an experienced expert in home renovations. You offer in-depth advice on renovation projects, "
                "especially those involving significant structural changes or additions to the house."
                "Always respond in Spanish unless otherwise indicated."
                "Use a maximum of 4 sentences, but keep the response as concise as possible."
                "Always end with 'otra pregunta' (another question) in the specified language."
            ),
            llm=self.llm
        )
    
    def materials_agent(self):
        return Agent(
            role='Materials expert',
            goal='Provide a detailed list of Materials used for the job',
            tools=[self.search_tool] + self.scrape_tools + self.pdf_tools,
            verbose=True,
            backstory=(
                "You are an experienced expert in construction. You offer in-depth advice of materials used."
                "Make a list using markdown with the materials and it's alternatives"
                "Always respond in Spanish unless otherwise indicated."
                "Use a maximum of 4 sentences, but keep the response as concise as possible."
            ),
            llm=self.llm
        )
    def herramientas_agent(self):
        return Agent(
            role='Tools expert',
            goal='Provide a detailed list of Tools used for the job',
            tools=[self.search_tool] + self.scrape_tools + self.pdf_tools,
            verbose=True,
            backstory=(
               "You are an experienced expert in construction. You offer in-depth advice of tools used."
                "Make a list using markdown with the tools and it's alternatives"
                "Always respond in Spanish unless otherwise indicated."
                "Use a maximum of 4 sentences, but keep the response as concise as possible."
            ),
            llm=self.llm
        )
    
    def cost_agent(self):
        return Agent(
            role='Cost determinator',
            goal='Based on the list of materials provide a table with the costs',
            tools=[self.search_tool] + self.scrape_tools + self.pdf_tools,
            verbose=True,
            backstory=(
               "You are a cost expert in construction. You offer in-depth estimation of the materials used."
                "Make a list using markdown with the cost of each material and it's alternatives"
                "Always respond in Spanish unless otherwise indicated."
                "Use a maximum of 4 sentences, but keep the response as concise as possible."
            ),
            llm=self.llm
        )
    

    ##------------------------------------TAKS------------------------------------
    def classify_project_task(self, question):
        return Task(
            description=f"Classify the following home improvement project: {question}. You can use synonym pages to find keywords similar to renovation or repair.",
            agent=self.planificator(),
            expected_output="Project classified as 'repair' or 'renovation'."
        )

    def provide_guidance_task(self, question, project_type):
        agent = self.repair_expert() if project_type == 'repair' else self.renovation_expert()
        return Task(
            description=f"Provide detailed guidance for this {project_type} project: {question}",
            agent=agent,
            expected_output=(
                "A complete guide for the home improvement project, including step-by-step instructions, "
                "required materials, estimated time and cost, and any safety precautions."
            )
        )
    ##------------------------------------CREATE CREW------------------------------------

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
            return f"Perdon, encontre un error en encontrar una solucion : {str(e)}"
