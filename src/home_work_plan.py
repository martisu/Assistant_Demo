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
#        self.scrape_tools = self.load_scrape_tools()
        self.pdf_tools = self.load_pdf_tools()
#        self.all_tools = [self.search_tool] + self.scrape_tools + self.pdf_tools
        self.context = {
            'guidance': None,
            'gather_info': None,
            'current_project': None,
            'project_type': None,
            'materials': [],
            'tools': [],
            'cost_estimation': None,
            'step_by_step_guide': None,
            'contractors': [],
            'conversation_history': []
        }
        
    def load_credentials(self, path):
        with open(path, "r") as stream:
            return yaml.safe_load(stream)

    def load_scrape_tools(self, source_type):
        """Load scraping tools based on the specified source type (either 'websites' or 'tools')."""
        file_path = f"data/sites/{source_type}.yaml"
        with open(file_path, "r") as file:
            resources = yaml.safe_load(file)

        scrape_tools = []
        for resource in resources['resources']:
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

    def image_agent(self):
        return

    def relevance_checker_agent(self):
        return Agent(
            role='Relevance Checker and Redirector',
            goal='Determine if a query is related to home improvement projects and provide a helpful response.',
            tools=[self.search_tool],
            verbose=True,
            backstory=(
                "You are an expert in home improvement projects with excellent communication skills. "
                "Your task is to determine if a question is about home repairs, renovations, or any other home improvement task. "
                "If the input is not related, gently redirect the conversation back to home improvement. "
                "Always respond in the language of the user. "
                "Provide a friendly explanation and suggest a related home improvement topic if possible."
            ),
            llm=self.llm
        )

    def planificator_agent(self):
        return Agent(
            role='Project Classifier',
            goal='Classify whether a home improvement project is a repair, a renovation, or undefined.',
            tools=[self.search_tool],
            verbose=True,
            backstory=(
                "You are an expert in classifying home improvement projects. "
                "Your task is to determine if a project is a **repair** (fixing or restoring something damaged), "
                "a **renovation** (improving or modernizing an existing feature), or **undefined** if it's not clear. "
                "Always respond in the language of the user unless otherwise indicated. "
                "If you can't determine the type, classify it as 'undefined'. "
                "Use a maximum of one line per response to keep it concise."
            ),
            llm=self.llm
        )
    
    def image_description_agent(self):
        return Agent(
            role='Damage Identifier',
            goal='Identify and describe any visible damage or areas requiring repair in an uploaded image related to home improvement projects.',
            tools=[self.image_recognition_tool],
            verbose=True,
            backstory=(
                "You are an expert in analyzing images of home interiors and exteriors for damage or repair needs. "
                "Your task is to examine the provided image and identify any parts that appear damaged, worn, or in need of repair. "
                "Generate a concise, clear description of the identified issues, including the location (e.g., ceiling, wall) and nature "
                "of the damage (e.g., cracks, water stains, loose tiles). Always respond in the language of the user unless otherwise indicated. "
                "If you cannot detect any visible damage, simply respond with 'No visible damage detected'. "
                "Use one to two sentences for each description to ensure clarity and brevity."
            ),
            llm=self.llm
        )


    def repair_agent(self):
        return Agent(
            role='Repair Expert',
            goal='Provide detailed guidance on home repair projects.',
            tools=[self.search_tool] + self.load_scrape_tools("websites") + self.pdf_tools,
            verbose=True,
            backstory=(
                "You are an experienced expert in home repairs. "
                "Your role is to provide clear and practical guidance on repair projects, "
                "focusing on fixes and maintenance tasks that do not require major structural changes. "
                "Always respond in the language of the user unless otherwise indicated. "
                "Use a maximum of four sentences to keep the response concise. "
                "Always end your response with 'another question?' in the specified language."
            ),
            llm=self.llm
        )

    def renovation_agent(self):
        return Agent(
            role='Renovation Expert',
            goal='Provide detailed guidance on home renovation projects.',
            tools=[self.search_tool] + self.pdf_tools,
            verbose=True,
            backstory=(
                "You are an experienced expert in home renovations. "
                "Your role is to provide in-depth advice on renovation projects, "
                "particularly those involving major structural changes or additions to the home. "
                "Always respond in the language of the user unless otherwise indicated. "
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
                "Always respond in language of the user unless otherwise indicated. "
                "Use a maximum of four sentences to keep the response concise."
            ),
            llm=self.llm
        )

    
    def tools_agent(self):
        return Agent(
            role='Tools Expert',
            goal='Based on the task context, provide a specific list of tools needed for the job.',
            tools=[self.search_tool]  + self.pdf_tools,
            verbose=True,
            backstory=(
            "You are an expert in selecting the right tools for specific construction tasks. "
            "You have access to a comprehensive list of tools scraped from reliable sources. "
            "Using the task context provided, filter and select only the tools required for the specific job. "
            "Exclude any materials or unrelated items. Provide the list in markdown format, "
            "including alternatives if available. Respond concisely in the language of the user."
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
                "Always respond in the language of the user unless otherwise indicated. "
                "Use a maximum of four sentences to keep the response concise."
            ),
            llm=self.llm
        )
    def guide_agent(self):
        return Agent(
            role='Step-by-Step Guide',
            goal='Provide detailed step-by-step instructions for any task.',
            tools=[self.search_tool] + self.pdf_tools,
            verbose=True,
            backstory=(
                "You are an expert guide. Your role is to break down complex tasks into clear, manageable steps."
                "Always ensure the instructions are simple and precise, adjusting based on the user's feedback."
                "Provide explanations for each step, but keep them concise."
                "Respond in the language of the user unless otherwise indicated."
            ),
            llm=self.llm
        )
    
    def contractor_search_agent(self):
        return Agent(
            role='Contractor Finder',
            goal='Search for contractors who can handle home improvement projects and provide contact details or links for budget estimation.',
            tools=[self.search_tool, *self.scrape_tools],
            verbose=True,
            backstory=(
                "You are an expert in finding reliable contractors for home improvement projects. "
                "Your role is to find contractors based on the user’s project description, preferably near their location. "
                "Search for relevant contractor listings, company websites, and review aggregators, and provide contact details or links where they can request a budget. "
                "Always respond in the language of the user unless otherwise indicated."
            ),
            llm=self.llm
        )
    
    def safety_agent(self):
        return Agent(
            role='Safety-Focused Task Guide',
            goal='Provide step-by-step instructions for tasks in a way that maximizes safety and minimizes the risk of accidents.',
            tools=[self.search_tool] + self.pdf_tools,
            verbose=True,
            backstory=(
                "You are a safety-focused expert responsible for guiding users through tasks with an emphasis on preventing accidents. "
                "Your role is to identify potential hazards and offer specific, precautionary steps to ensure safety. "
                "For each task, outline the required safety measures, such as protective gear, safety checks, or any specific warnings. "
                "Provide clear, detailed instructions, making sure to emphasize steps where caution is required. "
                "Respond in the language of the user unless otherwise specified, and adapt instructions based on the context and task complexity."
            ),
            llm=self.llm
        )

##------------------------------------TASKS------------------------------------

    def check_relevance_task(self, question):
        recent_history = self.context['conversation_history'][-5:]
        history_str = "\n".join([f"{entry['role']}: {entry['content']}" for entry in recent_history])

        return Task(
            description=(
                f"Considering the following recent conversation history:\n\n{history_str}\n\n"
                f"Analyze if the following query is related to home improvement projects: {question}\n"
                f"If it's related, respond with 'RELATED: ' followed by a brief confirmation.\n"
                f"If it's not related, respond with 'NOT RELATED: ' followed by a friendly message that:\n"
                f"1. Acknowledges the user's question\n"
                f"2. Gently reminds them that you're specialized in home improvement\n"
                f"3. Suggests a related home improvement topic they might be interested in\n"
                f"Ensure your response is in the same language as the user's query."
            ),
            agent=self.relevance_checker_agent(),
            expected_output="A decision of 'RELATED: ' or 'NOT RELATED: ' followed by an appropriate response."
        )
    def image_description_task(self, image):
        return Task(
            description=(
                "Examine the provided image and generate a detailed description of any visible damage or areas requiring repair. "
                "The description should include the specific location (e.g., ceiling, wall) and nature of the damage (e.g., cracks, water stains, loose tiles). "
                "If no damage is detected, respond with 'No visible damage detected'."
            ),
            agent=self.image_description_agent(),
            input=image,
            expected_output=(
                "A concise description of the damaged areas or issues in the image, e.g.,:\n\n"
                "- 'The ceiling shows water stains and cracks, likely due to leakage.'\n"
                "- 'Several tiles on the floor are loose and may need replacement.'"
            )
        )


    def planificator_task(self, question):
        return Task(
            description=f"Classify the following home improvement project: {question}. "
                        f"Determine if it's a repair, renovation, or if it's unclear (undefined).",
            agent=self.planificator_agent(),
            expected_output="Project classified as 'repair', 'renovation', or 'undefined'."
        )

    def gather_all_information(self, question):
        project_type = self.context['project_type']
        agent = self.repair_agent() if project_type == 'repair' else self.renovation_agent()
        return Task(
            description=(
                f"Assemble a comprehensive guide for this {project_type} project based on the user question: '{question}'. "
                f"Ensure the guidance covers each aspect in detail, making it user-friendly and actionable."
            ),
            agent=agent,
            expected_output=(
                "An all-inclusive project guide featuring: "
                "- Clear, step-by-step instructions from preparation to completion.\n"
                "- A list of all materials and tools required, specifying quantities and alternatives if needed.\n"
                "- Estimated project timeline with stages, highlighting potential delays or complex steps.\n"
                "- A breakdown of costs, including material, tools, and any recommended professional help.\n"
                "- Essential safety tips and precautions tailored to the project type.\n"
                "- Optional recommendations for design, quality assurance tips, and environmentally friendly practices."
            )
        )


    def materials_task(self, project_description):
        gather_info = self.context['gather_info']
        return Task(
            description=f"List the materials required for the following project: {project_description}. "
                        f"Consider additional info in context: {gather_info}. " 
                        f"It should only contain the MATERIALS, DO NOT add the TOOLS"
                        f"Include alternatives where applicable.",
            agent=self.materials_agent(),
            expected_output=(
                "A markdown list of materials and their alternatives, e.g.,:\n\n"
                "- **Material 1**: Description\n  - Alternative: Option 1\n  - Alternative: Option 2\n"
            )
    )

    def tools_task(self, project_description):
        gather_info = self.context['gather_info']
        materials = self.context['materials']
        return Task(
            description=f"List the tools required for the following project: {project_description}. "
                        f"Consider additional info in context: {gather_info}. " 
                        f"Consider materials in context: {materials}. " 
                        f"It should only contain the TOOLS, DO NOT add the MATERIALS",                        
            agent=self.tools_agent(),
            expected_output=(
                "A markdown list of tools and their alternatives, e.g.,:\n\n"
                "- **Tool 1**: Description\n  - Alternative: Option 1\n  - Alternative: Option 2\n"
            )
    )
 
    def cost_estimation_task(self, materials_list):
        gather_info = self.context['gather_info']
        materials = self.context['materials']
        tools = self.context['tools']
        return Task(
            description=f"Provide a detailed cost estimation for the following materials: {materials_list}. "
                        f"Consider additional info in context: {gather_info}. " 
                        f"Consider materials in context: {materials}. " 
                        f"Consider tools in context: {tools}. " 
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
        gather_info = self.context['gather_info']
        materials = self.context['materials']
        tools = self.context['tools']
        return Task(
            description=f"Provide detailed step-by-step instructions for the following repair or renovation process: {repair_or_renovation_process}. "
                        f"Consider additional info in context: {gather_info}. " 
                        f"Consider materials in context: {materials}. " 
                        f"Consider tools in context: {tools}. " 
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
    
    def contractor_search_task(self, project_description):
        gather_info = self.context['gather_info']
        materials = self.context['materials']
        tools = self.context['tools']
        return Task(
            description=(
                f"Search for contractors who specialize in the following project in the spedificated location: {project_description}. "
                f"Consider additional info in context: {gather_info}. " 
                f"Consider materials in context: {materials}. " 
                f"Consider tools in context: {tools}. "
                f"Provide contact details or links where the user can request a budget estimation. "
                f"Ensure the contractors are well-reviewed or reputable, if possible."
            ),
            agent=self.contractor_search_agent(),
            expected_output=(
                "A list of contractors with their contact information or website links, including details on how to request a budget."
            )
            # human_input=True
        )

    def safety_task(self, task_description):
        return Task(
            description=(
                f"Provide a careful, safety-focused guide for the following task: {task_description}. "
                f"The instructions should prioritize accident prevention by outlining each step in detail, highlighting any safety risks, "
                f"and suggesting appropriate protective measures or precautions. Emphasize where extra caution is needed."
            ),
            agent=self.safety_agent(),
            expected_output=(
                "A step-by-step safety guide with specific cautions, e.g.,:\n\n"
                "- **Step 1**: Identify the area where you will work and clear any obstacles.\n"
                "  - **Safety Tip**: Ensure the floor is dry to avoid slips.\n"
                "- **Step 2**: Gather necessary materials and wear protective gloves.\n"
                "  - **Safety Tip**: Use gloves rated for chemical handling if using cleaning agents.\n"
            )
        )


    ##------------------------------------CREATE CREW------------------------------------

    def get_response(self, question):
        try:
            self.context['conversation_history'].append({"role": "user", "content": question})

            # Step 0: Check relevance
            relevance_task = self.check_relevance_task(question)
            relevance_crew = Crew(
                agents=[relevance_task.agent],
                tasks=[relevance_task],
                verbose=True
            )
            relevance_result = relevance_crew.kickoff()

            if relevance_result.lower().startswith('not related:'):
                result = relevance_result.split(':', 1)[1].strip() 
            else:
                # Step 1: Classify the project (only if not already defined)
                if self.context['project_type'] is None or self.context['project_type'] == 'undefined':
                    classification_task = self.planificator_task(question)
                    classification_crew = Crew(
                        agents=[classification_task.agent],
                        tasks=[classification_task],
                        verbose=True
                    )
                    self.context['project_type'] = classification_crew.kickoff()
                    
#                    self.context['project_type'] = 'repair' if 'repair' in classification_result.lower() else 'renovation'

                    # Step 2: Provide guidance
                    guidance_task = self.gather_all_information(question)
                    guidance_crew = Crew(agents=[guidance_task.agent], tasks=[guidance_task], verbose=True)
                    self.context['gather_info'] = guidance_crew.kickoff()

                    # Step 3: List materials
                    materials_task = self.materials_task(question)
                    materials_crew = Crew(agents=[materials_task.agent], tasks=[materials_task], verbose=True)
                    self.context['materials'] = materials_crew.kickoff()

                    # Step 4: List tools
                    tools_task = self.tools_task(question)
                    tools_crew = Crew(agents=[tools_task.agent], tasks=[tools_task], verbose=True)
                    self.context['tools'] = tools_crew.kickoff()

                    # Step 5: Cost estimation
                    cost_task = self.cost_estimation_task(question)
                    cost_crew = Crew(agents=[cost_task.agent], tasks=[cost_task], verbose=True)
                    self.context['cost_estimation'] = cost_crew.kickoff()

                    # Step 6: Step-by-step guidance
                    guide_task = self.guide_task(question)
                    guide_crew = Crew(agents=[guide_task.agent], tasks=[guide_task], verbose=True)
                    self.context['step_by_step_guide'] = guide_crew.kickoff()

                    # Step 7: Contractor search
                    contractor_task = self.contractor_search_task(question) 
                    contractor_crew = Crew(agents=[contractor_task.agent], tasks=[contractor_task], verbose=True)
                    self.context['contractors'] = contractor_crew.kickoff()

                    # Create the main crew
#                    home_improvement_crew = Crew(
#                        agents=[gather_info.agent, 
#                                materials_task.agent, 
#                                tools_task.agent, 
#                                cost_task.agent, 
#                                guide_task.agent],
#
#                        tasks=[gather_info, 
#                               materials_task, 
#                               tools_task, 
#                               cost_task, 
#                               guide_task],
#                        verbose=2
#                    )            
#                    result = home_improvement_crew.kickoff()

                    result = (
                        f"**Project Guidance:**\n{self.context['guidance']}\n\n"
                        f"**Required Materials:**\n{self.context['materials']}\n\n"
                        f"**Required Tools:**\n{self.context['tools']}\n\n"
                        f"**Cost Estimation:**\n{self.context['cost_estimation']}\n\n"
                        f"**Step-by-Step Guide:**\n{self.context['step_by_step_guide']}\n\n"
                        f"**Recommended Contractors:**\n{self.context['contractors']}"
                    )

            self.context['conversation_history'].append({"role": "assistant", "content": result})

            return result
        
        except AttributeError as e:
            return f"Lo siento, parece que hay un problema con una de las herramientas o atributos: {str(e)}"
        except Exception as e:
            return f"Perdón, encontré un error al buscar una solución: {str(e)}"