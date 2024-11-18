from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool
#from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import ScrapeWebsiteTool
import yaml
import os

class CrewAIChatbot:
    def __init__(self, credentials_path):
        self.credentials = self.load_credentials(credentials_path)

        # Set OpenAI API key in environment variables
        os.environ["AZURE_API_KEY"] = self.credentials["AZURE_API_KEY"]
        os.environ["AZURE_API_BASE"] = self.credentials["AZURE_ENDPOINT"]
        os.environ["AZURE_API_VERSION"] = self.credentials["AZURE_API_VERSION"]
        
        self.llm = AzureChatOpenAI(
            deployment_name=self.credentials["MODEL_NAME"], 
            openai_api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_API_BASE"], 
            openai_api_version=os.environ["AZURE_API_VERSION"],
            openai_api_type="azure", 
            temperature=0.1

        )
        self.search_tool = DuckDuckGoSearchRun()
        self.pdf_tools = self.load_pdf_tools()
        
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
            'safety_guidance': None,
            'conversation_history': []
        }

    def load_pdf_tools(self):
        pdf_tools = []
        pdf_dir = "data/"
        
        tool_config = {
            "embedding_model": {
                "provider": "azure_openai",
                "config": {
                    "api_key": os.environ.get("AZURE_API_KEY"),
                    "azure_endpoint": os.environ.get("AZURE_API_BASE"),
                    "deployment_name": self.credentials.get("MODEL_EMBEDDING", "text-embedding-ada-002"),
                    "api_version": os.environ.get("AZURE_API_VERSION")  
                }
            }
        }
        
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, filename)
                try:
                    # Create PDFSearchTool with configuration
                    pdf_tool = PDFSearchTool(
                        file_path=pdf_path,
                        name=f"PDF_Reader_{filename}",
                        description=f"Use this tool to search and extract information from the PDF file {filename}",
                        max_chunks=5,
                        config=tool_config
                    )
                    pdf_tools.append(pdf_tool)
                except Exception as e:
                    print(f"Failed to load PDF tool for {filename}: {e}")

        return pdf_tools

    def load_credentials(self, path):
        with open(path, "r") as stream:
            return yaml.safe_load(stream)

    def reset_project(self):
        conversation_history = self.context.get('conversation_history', [])
        self.context = {
            'guidance': None,
            # 'gather_info': None,
            'current_project': None,
            'project_type': None,
            'materials': [],
            'tools': [],
            'cost_estimation': None,
            'step_by_step_guide': None,
            'contractors': [],
            'safety_guidance': None,
            'conversation_history': conversation_history,
        }

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
    ##------------------------------------AGENTS------------------------------------
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
                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise."
                "If you can't determine the type, classify it as 'undefined'. "
                "Use a maximum of one line per response to keep it concise."
                "Always respond in the language of the user. "
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
                "of the damage (e.g., cracks, water stains, loose tiles)."
                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise."
                "If you cannot detect any visible damage, simply respond with 'No visible damage detected'. "
                "Use one to two sentences for each description to ensure clarity and brevity."
            ),
            llm=self.llm
        )


    def repair_agent(self):
        return Agent(
            role='Repair Expert',
            goal='Provide detailed guidance on home repair projects.',
            tools=[self.search_tool] + self.pdf_tools,
            verbose=True,
            backstory=(
                "You are an experienced expert in home repairs. "
                "Your role is to provide clear and practical guidance on repair projects, "
                "focusing on fixes and maintenance tasks that do not require major structural changes. "
                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise."
                "Use a maximum of four sentences to keep the response concise. "
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
                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise."
                "Use a maximum of four sentences to keep the response concise. "
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
                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise."
                "Analyze if there is enough information to perform the task. "
                "Use a maximum of four sentences to keep the response concise."
            ),
            llm=self.llm
        )

    
    def tools_agent(self):
        return Agent(
            role='Tools Expert',
            goal='Based on the task context, provide a specific list of tools needed for the job.',
            tools=[self.search_tool],
            verbose=True,
            backstory=(
            "You are an expert in selecting the right tools for specific construction tasks. "
            "You have access to a comprehensive list of tools scraped from reliable sources. "
            "Using the task context provided, filter and select only the tools required for the specific job. "
            "Exclude any materials or unrelated items. Provide the list in markdown format, "
            "including alternatives if available. "
            "Analyze if there is enough information to perform the task. "
            "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise."
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
                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise."
                "Analyze if there is enough information to perform the task. "
                "Respond using the currency of the user's location if specified; otherwise, default to euros."

            ),
            llm=self.llm
        )
    def guide_agent(self):
        return Agent(
            role='Step-by-Step Guide',
            goal='Provide detailed step-by-step instructions for any task.',
            tools=[self.search_tool],
            verbose=True,
            backstory=(
                "You are an expert guide. Your role is to break down complex tasks into clear, manageable steps."
                "Always ensure the instructions are simple and precise, adjusting based on the user's feedback."
                "Provide explanations for each step, but keep them concise."
                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise." 
                "Analyze if there is enough information to perform the task. "
                "Respond using the currency of the user’s location if specified; otherwise, default to euros."
            ),
            llm=self.llm
        )
    
    def contractor_search_agent(self):
        return Agent(
            role='Contractor Finder',
            goal='Search for contractors who can handle home improvement projects and provide contact details or links for budget estimation.',
            tools=[self.search_tool],
            verbose=True,
            backstory=(
                "You are an expert in finding reliable contractors for home improvement projects. "
                "Your role is to find contractors based on the user’s project description, preferably near their location. "
                "Search for relevant contractor listings, company websites, and review aggregators, and provide contact details or links where they can request a budget. "
                "Analyze if there is enough information to perform the task. "
                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise."
            ),
            llm=self.llm
        )
    
    def safety_agent(self):
        return Agent(
            role='Safety-Focused Task Guide',
            goal='Provide step-by-step instructions for tasks in a way that maximizes safety and minimizes the risk of accidents.',
            tools=[self.search_tool],
            verbose=True,
            backstory=(
                "You are a safety-focused expert responsible for guiding users through tasks with an emphasis on preventing accidents. "
                "Your role is to identify potential hazards and offer specific, precautionary steps to ensure safety. "
                "For each task, outline the required safety measures, such as protective gear, safety checks, or any specific warnings. "
                "Provide clear, detailed instructions, making sure to emphasize steps where caution is required. "
                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise."
                "Analyze if there is enough information to perform the task. "
                "adapt instructions based on the context and task complexity."
            ),
            llm=self.llm
        )

    def presentation_agent(self):
        return Agent(
            role="Presentation Expert",
            goal="Compose a clear, well-structured response with all gathered project details in the user's language.",
            tools=[],  # No external tools are required for presentation composition
            verbose=True,
            backstory=(
                "You are a presentation expert responsible for assembling and presenting all relevant project information "
                "in a clear and structured format. Your role is to create a coherent response that includes project guidance, "
                "required materials, tools, cost estimation, step-by-step guide, and recommended contractors. "
                "Ensure the response is understandable, visually organized, and presented in the user's language."
            ),
            llm=self.llm
        )


##------------------------------------TASKS------------------------------------

    def check_relevance_task(self, question):
        recent_history = self.context['conversation_history'][-50:]
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
    
    def image_description_task(self, image=None):
        """
        Task for analyzing an uploaded image. If no image is provided, the task skips analysis.
        """
        if image:
            return Task(
                description=(
                    "Examine the provided image and generate a detailed description of any visible damage or areas requiring repair. "
                    "The description should include the specific location (e.g., ceiling, wall) and nature of the damage (e.g., cracks, water stains, loose tiles). "
                    "If no damage is detected, respond with 'No visible damage detected'."
                ),
                agent=self.image_agent(),
                input=image,
                expected_output=(
                    "A concise description of the damaged areas or issues in the image, e.g.,:\n\n"
                    "- 'The ceiling shows water stains and cracks, likely due to leakage.'\n"
                    "- 'Several tiles on the floor are loose and may need replacement.'"
                )
            )
        else:
            return None  # No image provided, skip the task


    def planificator_task(self, question):
        recent_history = self.context['conversation_history'][-50:]
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"Classify the following home improvement project: {question}. "
                        f"Determine if it's a repair, renovation, or if it's unclear (undefined).",
            agent=self.planificator_agent(),
            expected_output="Project classified as 'repair', 'renovation', or 'undefined'."
        )

    def gather_all_information(self, question):
        recent_history = self.context['conversation_history'][-50:]
        project_type = self.context['project_type']
        agent = self.repair_agent() if project_type == 'repair' else self.renovation_agent()
        return Task(
            description=(
                f"Consider the conversation history: {recent_history}."
                f"Based on the project type ({project_type}) and the user's question '{question}', assemble a comprehensive guide. "
                f"Analyze the provided question and context to determine if additional information is required from the user to create a complete project guide. Make sure to know all relevant details."
                f"If additional information is needed, specify it clearly by starting your response with 'question:'."

            ),
            agent=agent,
            expected_output=(
                "If more information from the user is required, answer 'question:' followed by a clear and specific question in the language of the user. For example:\n"
                "- 'What is the size of the area to be repaired?'\n"
                "- 'What type of materials would you like to use for the renovation?'\n"
                "If no additional information is needed, provide a detailed project guide including:\n"
                "- Clear, step-by-step instructions from preparation to completion.\n"
                "- A list of all materials and tools required, specifying quantities and alternatives if needed.\n"
                "- Estimated project timeline with stages, highlighting potential delays or complex steps.\n"
                "- A breakdown of costs, including material, tools, and any recommended professional help.\n"
                "- Essential safety tips and precautions tailored to the project type.\n"
                "- Optional recommendations for design, quality assurance tips, and environmentally friendly practices."
            )
        )


    def materials_task(self, project_description):
        recent_history = self.context['conversation_history'][-50:]
        # gather_info = self.context['gather_info']
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"List the materials required for the following project: {project_description}. "
                        # f"Consider additional info in context: {gather_info}. "
                        f"The response should only contain the MATERIALS and must NOT include the TOOLS. "
                        f"Include alternatives where applicable. "
                        f"Analyze if there is enough information to perform the task and provide an appropriate response and make sure to know all relevant details"
                        f"If information is missing, generate a specific question for the user.",
            agent=self.materials_agent(),
            expected_output=(
                "If more information from the user is required, answer 'question:' followed by a clear and specific question in the language of the user. For example:\n"
                "- 'What type of materials would you prefer for this project?'\n"
                "- 'What is the size or scope of the project to estimate the quantity of materials needed?'\n"
                "If no more information from the user is needed, answer with a markdown list of materials with quantities and alternatives, e.g.:\n\n"
                "- **Material 1**: High-quality cement\n"
                "  - Quantity: 10 kg\n"
                "  - Alternative: Eco-friendly cement (8 kg)\n"
                "- **Material 2**: Paint (white)\n"
                "  - Quantity: 2 liters\n"
                "  - Alternative: Matte finish paint (2 liters)\n"
            )
    )

    def tools_task(self, project_description):
        recent_history = self.context['conversation_history'][-50:]
        # gather_info = self.context['gather_info']
        materials = self.context['materials']
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"List the tools required for the following project: {project_description}. "
                        # f"Consider any additional information provided in context: {gather_info}. "
                        f"Consider materials provided in context: {materials}. "
                        f"The response should only contain the TOOLS and must NOT include the MATERIALS. "
                        f"Analyze if there is enough information to perform the task effectively. Make sure to know all relevant details "
                        f"If key information is missing, generate a specific question to ask the user to gather the necessary details.",        
            agent=self.tools_agent(),
            expected_output=(
                "If more information from the user is required, answer 'question:' followed by a clear and specific question in the language of the user. For example:\n"
                "- 'Could you specify if you have any preferred tools for this project?'\n"
                "- 'Do you need any specialized tools to work with the materials you have chosen?'\n"
                "If no additional information from the user is needed, answer with a markdown list of tools and their alternatives, e.g.:\n\n"
                "- **Tool 1**: Electric drill\n"
                "  - Alternative: Manual drill\n"
                "- **Tool 2**: Screwdriver (Phillips)\n"
                "  - Alternative: Flathead screwdriver\n"
            )
    )
 
    def cost_estimation_task(self, materials_list):
        recent_history = self.context['conversation_history'][-50:]
        # gather_info = self.context['gather_info']
        materials = self.context['materials']
        tools = self.context['tools']
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"Provide a detailed cost estimation for the following materials: {materials_list}. "
                        # f"Consider any additional info provided in context: {gather_info}. "
                        f"Consider materials provided in context: {materials}. "
                        f"Consider tools provided in context: {tools}. "
                        f"Include costs for alternatives where applicable. "
                        f"Analyze if there is enough information to perform the task. Make sure to know all relevant details "
                        f"If key information is missing, generate a specific question to ask the user to gather the necessary details.",
            agent=self.cost_agent(),
            expected_output=(
                "If more information from the user is required, answer 'question:' followed by a clear and specific question in the language of the user. For example:\n"
                "- 'Could you specify the type or quality of materials for accurate cost estimation?'\n"
                "- 'Do you have a budget range for each material or tool?'\n"
                "If no additional information from the user is needed, answer with a markdown table listing materials and their costs, including alternatives, e.g.:\n\n"
                "| Material        | Cost  | Alternatives                       |\n"
                "|----------------|-------|------------------------------------|\n"
                "| Material 1     | $10   | Alternative 1 ($8), Alt 2 ($12)   |\n"
                "| Material 2     | $15   | Alternative 1 ($12), Alt 2 ($18)  |\n"
            )
        )
    
    def guide_task(self, repair_or_renovation_process):
        recent_history = self.context['conversation_history'][-50:]
        # gather_info = self.context['gather_info']
        materials = self.context['materials']
        tools = self.context['tools']
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"Provide detailed step-by-step instructions for the following repair or renovation process: {repair_or_renovation_process}. "
                        # f"Consider any additional info provided in context: {gather_info}. "
                        f"Consider materials provided in context: {materials}. "
                        f"Consider tools provided in context: {tools}. "
                        f"Ensure that the steps are easy to follow and comprehensive, covering all necessary tools, materials, and safety precautions. "
                        f"Analyze if there is enough information to perform the task. Make sure to know all relevant details."
                        f"If key information is missing, generate a specific question to ask the user to gather the necessary details.",
            agent=self.guide_agent(),
            expected_output=(
                "If more information from the user is required, answer 'question:' followed by a clear and specific question in the language of the user. For example:\n"
                "- 'What specific steps do you want to include in the process?'\n"
                "- 'Are there any special materials or safety considerations for this task?'\n"
                "If no additional information from the user is needed, answer with a list of detailed steps for the repair or renovation process. For example:\n\n"
                "1. Identify the scope of the repair or renovation.\n"
                "2. Gather all necessary tools and materials.\n"
                "3. Prepare the work area to ensure safety and efficiency.\n"
                "4. Step-by-step breakdown of the actual work (e.g., removing old materials, installing new ones).\n"
                "5. Final touches and clean-up instructions.\n"
            )
        )
    
    def contractor_search_task(self, project_description):
        recent_history = self.context['conversation_history'][-50:]
        # gather_info = self.context['gather_info']
        materials = self.context['materials']
        tools = self.context['tools']
        return Task(
            description=(
                f"Consider the conversation history: {recent_history}."
                f"Search for contractors who specialize in the following project in the specified location or nearby. "
                # f"Consider any additional info provided in context: {gather_info}. "
                f"Consider materials provided in context: {materials}. "
                f"Consider tools provided in context: {tools}. "
                f"Provide contact details or links where the user can request a budget estimation. "
                f"Ensure the contractors are well-reviewed or reputable, if possible. "
                f"Analyze if there is enough information to perform the task, including location details. Make sure to know all relevant details."
                f"If key information is missing, generate a specific question to ask the user to gather the necessary details."
            ),
            agent=self.contractor_search_agent(),
            expected_output=(
            "If more information from the user is required, answer 'question:' followed by a clear and specific question in the language of the user. For example:\n"
            "- 'Could you specify the location for the project so I can find local contractors?'\n"
            "- 'Are there any specific requirements or certifications needed for the contractors?'\n"
            "If no additional information from the user is needed, answer with a list of contractors with their contact information or website links, including details on how to request a budget estimation. For example:\n\n"
            "- **Contractor 1**: ABC Renovations\n"
            "  - Contact: (123) 456-7890\n"
            "  - Website: [www.abcrenovations.com](http://www.abcrenovations.com)\n"
            "- **Contractor 2**: Home Fix Pros\n"
            "  - Contact: (987) 654-3210\n"
            "  - Website: [www.homefixpros.com](http://www.homefixpros.com)\n"
            )
#            human_input=True
        )

    def safety_task(self, task_description):
        recent_history = self.context['conversation_history'][-10:]
        # gather_info = self.context['gather_info']
        materials = self.context['materials']
        tools = self.context['tools']
        step_by_step_guide = self.context['step_by_step_guide']
        return Task(
            description=(
                f"Consider the conversation history: {recent_history}."
                f"Provide a careful, safety-focused guide for the following task: {task_description}. "
                f"The instructions should prioritize accident prevention by outlining each step in detail, highlighting any safety risks,"
                f"and suggesting appropriate protective measures or precautions. Emphasize where extra caution is needed."
                f"Consider the question of the user: {task_description}."
                # f"Consider additional info in context: {gather_info}. " 
                f"Consider materials in context: {materials}. " 
                f"Consider tools in context: {tools}. "
                f"Consider step by step guide in context: {step_by_step_guide}. " 
                f"Analyze if there is enough information to perform the task. Make sure to know all relevant details."
            ),
            agent=self.safety_agent(),
            expected_output=(
                "If more information from the user is required, answer 'question:' followed by the text asking for the missing information needed to provide a complete response in the language of the user."
                "If no more information from the user is needed, respond with a concise step-by-step safety guide. Include the following considerations:\n"
                "- Key steps to perform the task safely."
                "- Specific risks associated with each step."
                "- Protective measures to mitigate risks."
                "- Areas where extra caution is needed."
                "If additional information is required, respond with 'question:' followed by a clear request for the missing details."
                "(Text indicating the missing information needed to provide a complete response, if necessary.)"
            )
        )

    def presentation_task(self, task_description):
        recent_history = self.context['conversation_history'][-50:]
        # gather_info = self.context['gather_info']
        materials = self.context['materials']
        tools = self.context['tools']
        step_by_step_guide = self.context['step_by_step_guide']
        contractors = self.context['contractors']
        cost_estimation = self.context['cost_estimation']
        safety_guidance = self.context['safety_guidance']

        return Task(
            description=(
                f"Consider the conversation history: {recent_history}."
                f"Compose a final, well-structured response with the collected project details based on the user's question."
                f"Include project guidance, required materials, tools, cost estimation, step-by-step guide, and recommended contractors. "
                f"Ensure that the response is visually clear, well-organized, and presented in the user's language. "
                f"If translation is necessary, adapt the response to the language detected in the user's question."
                f"Consider the question of the user: {task_description}."
                # f"Consider additional info in context: {gather_info}. " 
                f"Consider materials in context: {materials}. (Also missing ininformation if there is indicated)." 
                f"Consider tools in context: {tools}. (Also missing ininformation if there is indicated)."
                f"Consider step by step guide in context: {step_by_step_guide}. (Also missing ininformation if there is indicated)."
                f"Consider contractors in context: {contractors}. (Also missing ininformation if there is indicated)."
                f"Consider cost estimation in context: {cost_estimation}. (Also missing ininformation if there is indicated)."
                f"Consider safty guidance notes in context: {safety_guidance}. (Also missing ininformation if there is indicated)."
            ),
            agent=self.presentation_agent(),
            expected_output=(
                "If all information is complete, provide a structured response that includes the following:\n"
                "- Materials.\n"
                "- Tools.\n"
                "- Step by step guide.\n"
                "- Contractors information.\n"
                "- Cost estimation.\n"
                "- Safty guidance notes.\n"
                "- Questions asking for missing information if there is missing informationor.\n"
                "\n"
                "If information is missing, respond with a list of specific clarifying questions for the user to complete the missing details.\n"
                "For example:\n"
                "- What materials are required for the project?\n"
                "- What tools are necessary to complete the job?\n"
                "- Provide a step-by-step guide for the task.\n"
            )
        )

    ##------------------------------------CREATE CREW------------------------------------

    def get_response(self, question):
        try:
            result = None
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
                    
                        self.context['project_type'] = 'repair' if 'repair' in self.context['project_type'].lower() else 'renovation'

                    # Step 2: Provide guidance
                    # if result is None and self.context['gather_info'] is None:
                    #     guidance_task = self.gather_all_information(question)
                    #     guidance_crew = Crew(agents=[guidance_task.agent], tasks=[guidance_task], verbose=True)
                    #     self.context['gather_info'] = guidance_crew.kickoff()
                    #     if self.context['gather_info'].lower().startswith('question:'):
                    #         result = self.context['gather_info'].split(':', 1)[1].strip()
                    #         self.context['gather_info'] = None

                    # Step 3: List materials
                    if result is None and not self.context['materials']:
                        materials_task = self.materials_task(question)
                        materials_crew = Crew(agents=[materials_task.agent], tasks=[materials_task], verbose=True)
                        self.context['materials'] = materials_crew.kickoff()
                        if self.context['materials'].lower().startswith('question:'):
                            result = self.context['materials'].split(':', 1)[1].strip()
                            self.context['materials'] = []

                    # Step 4: List tools
                    if result is None and not self.context['tools']:
                        tools_task = self.tools_task(question)
                        tools_crew = Crew(agents=[tools_task.agent], tasks=[tools_task], verbose=True)
                        self.context['tools'] = tools_crew.kickoff()
                        if self.context['tools'].lower().startswith('question:'):
                            result = self.context['tools'].split(':', 1)[1].strip()
                            self.context['tools'] = []

                    # Step 5: Cost estimation
                    if result is None and self.context['cost_estimation'] is None:
                        cost_task = self.cost_estimation_task(question)
                        cost_crew = Crew(agents=[cost_task.agent], tasks=[cost_task], verbose=True)
                        self.context['cost_estimation'] = cost_crew.kickoff()
                        if self.context['cost_estimation'].lower().startswith('question:'):
                            result = self.context['cost_estimation'].split(':', 1)[1].strip()
                            self.context['cost_estimation'] = None


                    # Step 6: Step-by-step guidance
                    if result is None and self.context['step_by_step_guide'] is None:
                        guide_task = self.guide_task(question)
                        guide_crew = Crew(agents=[guide_task.agent], tasks=[guide_task], verbose=True)
                        self.context['step_by_step_guide'] = guide_crew.kickoff()
                        if self.context['step_by_step_guide'].lower().startswith('question:'):
                            result = self.context['step_by_step_guide'].split(':', 1)[1].strip()
                            self.context['step_by_step_guide'] = None

                    # Step 7: Contractor search
                    if result is None and not self.context['contractors']:
                        contractor_task = self.contractor_search_task(question) 
                        contractor_crew = Crew(agents=[contractor_task.agent], tasks=[contractor_task], verbose=True)
                        self.context['contractors'] = contractor_crew.kickoff()
                        if self.context['contractors'].lower().startswith('question:'):
                            result = self.context['contractors'].split(':', 1)[1].strip()
                            self.context['contractors'] = []


                    # Step 8: Safety guidance
                    if result is None and self.context['safety_guidance'] is None:
                        safety_task = self.safety_task(question)
                        safety_crew = Crew(agents=[safety_task.agent], tasks=[safety_task], verbose=True)
                        self.context['safety_guidance'] = safety_crew.kickoff()
                        if self.context['safety_guidance'].lower().startswith('question:'):
                            result = self.context['safety_guidance'].split(':', 1)[1].strip()
                            self.context['safety_guidance'] = None

                    # Presentation
                    if result is None:
                        presentation_task = self.presentation_task(question)
                        presentation_crew = Crew(agents=[presentation_task.agent], tasks=[presentation_task], verbose=True)
                        result = presentation_crew.kickoff()

                        # Reset project
                        self.reset_project()

            self.context['conversation_history'].append({"role": "assistant", "content": result})

            return result
        
        except AttributeError as e:
            return f"Lo siento, parece que hay un problema con una de las herramientas o atributos: {str(e)}"
        except Exception as e:
            return f"Perdón, encontré un error al buscar una solución: {str(e)}"
        


