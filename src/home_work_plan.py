from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import ScrapeWebsiteTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
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
        self.all_tools = [self.search_tool] + self.scrape_tools + self.pdf_tools
        
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
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, filename)
                loader = PyPDFLoader(pdf_path)
                documents = loader.load_and_split(text_splitter)
                
                # Limit the number of chunks to reduce token count
                max_chunks = 5
                limited_documents = documents[:max_chunks]
                
                pdf_tool = Tool(
                    name=f"PDF_Reader_{filename}",
                    func=lambda docs=limited_documents: "\n".join([doc.page_content for doc in docs]),
                    description=f"Use this tool to read and extract information from the PDF file {filename}"
                )
                pdf_tools.append(pdf_tool)
        return pdf_tools

    ##------------------------------------AGENTS------------------------------------
    def planificator_agent(self):
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

    def repair_agent(self):
        return Agent(
            role='Repair Expert',
            goal='Provide detailed guidance on home repair projects.',
            # tools=[self.search_tool] +self.scrape_tools + self.pdf_tools,
            tools=[self.search_tool] ,
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

    def renovation_agent(self):
        return Agent(
            role='Renovation Expert',
            goal='Provide detailed guidance on home renovation projects.',
            # tools=[self.search_tool] + self.pdf_tools,
            tools=[self.search_tool] ,
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
            tools=[self.search_tool],
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
            # tools=[self.search_tool] + self.pdf_tools,
            tools=[self.search_tool] ,
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
            tools=[self.search_tool],
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
    def guide_agent(self):
        return Agent(
            role='Step-by-Step Guide',
            goal='Provide detailed step-by-step instructions for any task.',
            # tools=[self.search_tool] + self.pdf_tools,
            tools=[self.search_tool],
            verbose=True,
            backstory=(
                "You are an expert guide. Your role is to break down complex tasks into clear, manageable steps."
                "Always ensure the instructions are simple and precise, adjusting based on the user's feedback."
                "Provide explanations for each step, but keep them concise."
                "Respond in Spanish unless otherwise indicated."
            ),
            llm=self.llm
        )


##------------------------------------TASKS------------------------------------

    def planificator_task(self, question):
        return Task(
            description=f"Classify the following home improvement project: {question}. "
                        f"You can use synonym pages to find keywords similar to renovation or repair.",
            agent=self.planificator_agent(),
            expected_output="Project classified as 'repair' or 'renovation'.",
            human_input=True

        )
    
    def provide_guidance_task(self, question, project_type):
        agent = self.repair_agent() if project_type == 'repair' else self.renovation_agent()
        return Task(
            description=f"Provide detailed guidance for this {project_type} project: {question}.",
            agent=agent,
            expected_output=(
                "A complete guide for the home improvement project, including step-by-step instructions, "
                "required materials, estimated time and cost, and any safety precautions."
            ),
            human_input=True
        )

    def materials_task(self, project_description):
        return Task(
            description=f"List the materials required for the following project: {project_description}. "
                        f"It should only contain the MATERIALS, DO NOT add the TOOLS"
                        f"Include alternatives where applicable.",
            agent=self.materials_agent(),
            expected_output=(
                "A markdown list of materials and their alternatives, e.g.,:\n\n"
                "- **Material 1**: Description\n  - Alternative: Option 1\n  - Alternative: Option 2\n"
            )
        )

    def tools_task(self, project_description):
        return Task(
            description=f"List the tools required for the following project: {project_description}. "
                        f"It should only contain the TOOLS, DO NOT add the MATERIALS",
            agent=self.tools_agent(),
            expected_output=(
                "A markdown list of tools and their alternatives, e.g.,:\n\n"
                "- **Tool 1**: Description\n  - Alternative: Option 1\n  - Alternative: Option 2\n"
            )
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
            )
        )
    def guide_task(self, repair_or_renovation_process):
        return Task(
            description=f"Provide detailed step-by-step instructions for the following repair or renovation process: {repair_or_renovation_process}. "
                        f"Ensure that the steps are easy to follow and comprehensive, covering all necessary tools, materials, and safety precautions.",
            agent=self.guide_agent(),
            expected_output=(
                "A list of detailed steps for the repair or renovation process. For example:\n\n"
                "1. Identify the scope of the repair or renovation.\n"
                "2. Gather all necessary tools and materials.\n"
                "3. Prepare the work area to ensure safety and efficiency.\n"
                "4. Step-by-step breakdown of the actual work (e.g., removing old materials, installing new ones).\n"
                "5. Final touches and clean-up instructions.\n"
            )
        )


    ##------------------------------------CREATE CREW------------------------------------

    def get_response(self, question):
        try:
            # Step 1: Classify the project
            classification_task = self.planificator_task(question)
            classification_crew = Crew(
                agents=[classification_task.agent],
                tasks=[classification_task],
                verbose=True,
                planning = True,
                memory = True
            )
            result = classification_crew.kickoff()
            project_type = 'repair' if 'repair' in result.lower() else 'renovation'

            # Step 2: Provide guidance
            guidance_task = self.provide_guidance_task(question, project_type)
            
            # Step 3: List materials
            materials_task = self.materials_task(question)
            
            # Step 4: List tools
            tools_task = self.tools_task(question)
            
            # Step 5: Cost estimation
            cost_task = self.cost_estimation_task(question)

            # Step 6: Step-by-step guidance
            guide_task = self.guide_task(question)

            # Create the main crew
            home_improvement_crew = Crew(
                agents=[guidance_task.agent, materials_task.agent, tools_task.agent, cost_task.agent, guide_task.agent],
                tasks=[guidance_task, materials_task, tools_task, cost_task, guide_task],
                verbose=True,
                planning = True,
                memory = True
            )
            
            result = home_improvement_crew.kickoff()

            return result
        except AttributeError as e:
            return f"Lo siento, parece que hay un problema con una de las herramientas o atributos: {str(e)}"
        except Exception as e:
            return f"Perdón, encontré un error al buscar una solución: {str(e)}"