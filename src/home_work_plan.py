
from crewai import Agent, Task, Crew, Process
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


from crewai_tools import ScrapeWebsiteTool
import yaml
import os

class CrewAIChatbot:

    HISTORY_LIMIT = 30  # Length of the history to consider

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
        self.wrapper = DuckDuckGoSearchAPIWrapper(max_results=2 )
        self.search_tool = DuckDuckGoSearchRun(api_wrapper =self.wrapper, source = "text", backend = "lite" )
        self.pdf_tools = self.load_pdf_tools()
        
        self.context = {
            'guidance': None,
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
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, filename)
                loader = PyPDFLoader(pdf_path)
                
                # Load and split the PDF
                try:
                    documents = loader.load_and_split(text_splitter)
                    print(f"Documents loaded for {filename}: {documents[:5]}")  # Debugging
                    
                    # Handle documents properly based on their structure
                    if documents and isinstance(documents[0], str):
                        # Handle case where documents are plain strings
                        limited_documents = documents[:5]
                        pdf_tool = Tool(
                            name=f"PDF_Reader_{filename}",
                            func=lambda docs=limited_documents: "\n".join(docs),
                            description=f"Use this tool to read and extract information from the PDF file {filename}"
                        )
                    elif documents and hasattr(documents[0], "page_content"):
                        # Handle case where documents are objects with 'page_content'
                        limited_documents = documents[:5]
                        pdf_tool = Tool(
                            name=f"PDF_Reader_{filename}",
                            func=lambda docs=limited_documents: "\n".join([doc.page_content for doc in docs]),
                            description=f"Use this tool to read and extract information from the PDF file {filename}"
                        )
                    else:
                        print(f"Unexpected structure for documents in {filename}: {documents}")
                        continue  # Skip if structure is not as expected

                    pdf_tools.append(pdf_tool)
                
                except Exception as e:
                    print(f"Error loading or splitting PDF {filename}: {e}")
                    continue  # Skip problematic PDFs
        
        return pdf_tools
    
    def cost_search(self, country_filter="Spain"):
        """
        Search within the 'Costs' section of the YAML file and filter by country if provided.
        Falls back to direct DuckDuckGo search if country not in YAML.
        
        Args:
            country_filter (str, optional): The country to limit the search to (e.g., 'Spain').
        
        Returns:
            dict: A dictionary of URLs and their respective search results or an error message.
        """
        try:
            # Step 1: Load YAML Data
            with open("data/sites/cost.yaml", "r") as file:
                data = yaml.safe_load(file)
            
            # Step 2: Check if country exists in YAML
            available_countries = [entry.get("country") for entry in data]
            
            # If country not in YAML, do direct DuckDuckGo search
            if country_filter not in available_countries:
                query = f"construction materials cost prices {country_filter} hardware store building supplies"
                direct_result = self.search_tool.run(query)
                return {
                    "direct_search": {
                        "country": country_filter,
                        "search_type": "direct",
                        "results": direct_result
                    }
                }
            
            # Step 3: Filter by country if it exists in YAML
            costs_data = [entry for entry in data if entry.get("country") == country_filter]
            
            # Step 4: Extract links from the section
            search_results = {}
            for entry in costs_data:
                country = entry.get("country")
                stores = entry.get("stores", [])
                for store in stores:
                    name = store.get("name")
                    link = store.get("link")
                    if link:
                        # Perform the search
                        query = f"site:{link}"  # Restrict search to the specific store link
                        result = self.search_tool.run(query)
                        search_results[f"{country} - {name}"] = result
            
            # Return the search results
            return search_results

        except Exception as e:
            return f"Error occurred: {str(e)}"



    def load_credentials(self, path):
        with open(path, "r") as stream:
            return yaml.safe_load(stream)

    def reset_project(self):
        conversation_history = self.context.get('conversation_history', [])
        self.context = {
            'guidance': None,
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
    def relevance_agent(self):
        return Agent(
            role='Relevance Checker and Redirector',
            goal='Determine if a query is related to home improvement projects and provide a helpful response.',
            tools=[],
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
            tools=[],
            verbose=True,
            backstory=(
                "You are an expert in classifying home improvement projects. "
                "Your task is to determine if a project is a **repair** (fixing or restoring something damaged), "
                "a **renovation** (improving or modernizing an existing feature), or **undefined** if it's not clear. "
                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise. "
                "If you can't determine the type, classify it as 'undefined'. "
                "Use a maximum of one line per response to keep it concise. "
                "Always respond in the language of the user."
            ),
            llm=self.llm
        )
    
    def questions_agent(self):
        return Agent(
            role='Information Gathering Specialist',
            goal='Ask targeted questions to collect all necessary details for a home improvement project.',
            tools=[],
            verbose=True,
            backstory=(
                "Your role is to gather all relevant information from the user to ensure the project is clearly defined and well-prepared. "
                "Acknowledge user responses dynamically and use them to adapt your follow-up questions. For instance, if the user provides dimensions, thank them and ask about budget next. "
                "Ensure your inquiries are clear, concise, and phrased in a professional yet approachable tone. "
                "If the user doesn’t have certain details, offer reassurance, and explain how you can help later in the process."
                "Always respond in the language of the user."
            ),
            llm=self.llm
        )

    def repair_agent(self):
        return Agent(
            role='Repair Expert',
            goal='Provide detailed guidance on home repair projects.',
            tools=[self.search_tool],
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
                "You are an expert in forecasting materials and determining the required quantity of each material, provided the user's instructions are sufficient."
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
            goal='Provide cost estimations for materials, considering the user’s location and preferred currency.',
            tools=[self.search_tool],
            verbose=True,
            backstory=(
                "You are a cost expert in construction. "
                "Your role is to provide cost estimations for materials or tools, converting them into the currency based on the user's location. "
                "Default to euros (€) if the user’s location is not specified. "
                "Perform a targeted search for pricing data and ensure clarity in the response. "
                "Provide approximate unit prices in the user’s currency or a specified currency. "
                "Avoid including unrelated context or general market trends."
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
    
    def scheduler_agent(self):
        return Agent(
            role='Project Scheduler',
            goal='Create a detailed schedule for home improvement projects based on user-provided constraints and priorities.',
            tools=[],
            verbose=True,
            backstory=(
                "You are a project scheduling expert with experience in construction and home improvement tasks. "
                "Your task is to organize project steps into a logical order and assign estimated durations. "
                "Take into account task dependencies, resource availability, and deadlines provided by the user. "
                "If constraints or resources are unclear, ask specific questions to clarify. "
                "Always ensure the schedule is realistic, and recommend adjustments if necessary to meet deadlines. "
                "Present the schedule in a structured markdown table or a clear step-by-step format."
            ),
            llm=self.llm
        )


    def presentation_agent(self):
        return Agent(
            role="Presentation Expert",
            goal="Compose a clear, well-structured response with all gathered project details in the user's language.",
            tools=[],  
            verbose=True,
            backstory=(
                "You are a presentation expert responsible for assembling and presenting all project information "
                "in a clear and structured format. Your role is to create a coherent response that includes project guidance, "
                "required materials, tools, cost estimation, step-by-step guide, and recommended contractors. "
                "Ensure the response is understandable, visually organized, and presented in the user's language."
            ),
            llm=self.llm
        )


##------------------------------------TASKS------------------------------------

    def check_relevance_task(self, question):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
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
            agent=self.relevance_agent(),
            expected_output="A decision of 'RELATED: ' or 'NOT RELATED: ' followed by an appropriate response."
        )

    def planificator_task(self, question):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"Classify the following home improvement project: {question}. "
                        f"Determine if it's a repair, renovation, or if it's unclear (undefined).",
            agent=self.planificator_agent(),
            expected_output="Project classified as 'repair', 'renovation', or 'undefined'."
        )

    def questions_task(self, question):
        # Function to sanitize input and prevent flagged terms
        def sanitize_text(text):
            restricted_terms = ["bypass", "hack", "exploit", "jailbreak"]
            for term in restricted_terms:
                text = text.replace(term, "[restricted]")
            return text

        # Sanitize inputs and history
        recent_history = [
            sanitize_text(entry["content"]) for entry in self.context["conversation_history"][-self.HISTORY_LIMIT:]
        ]
        sanitized_question = sanitize_text(question)
        project_type = self.context["project_type"]

        # Define the task
        return Task(
            description=(
                f"Analyze the sanitized conversation history: {recent_history}, the project type ({project_type}), "
                f"and the user's question '{sanitized_question}'. Your task is to gather all missing details needed to fully define the project. "
                "Based on the context and user input, determine which details are still unclear or incomplete. "
                "Ask specific, clear, and professional questions to fill these gaps, ensuring the user feels supported throughout the process. "
                "Adapt your questions to the project type—repair or renovation—and prioritize gathering details such as:\n"
                "- The dimensions or size of the project area.\n"
                "- The location of the project (e.g., indoor or outdoor, city for contractors).\n"
                "- The budget and material preferences (economical, premium, or sustainable).\n"
                "- Whether the user has any tools or materials available to use.\n"
                "- The desired timeline and urgency of the project.\n"
                "If the user lacks certain information, offer reassurance, and frame your questions in a way that keeps the process collaborative and stress-free."
            ),
            agent=self.questions_agent(),
            expected_output=(
                "If additional information is needed, start your response with 'question:' followed by specific and user-friendly questions. Examples include:\n"
                "- 'What are the dimensions of the area to be worked on?'\n"
                "- 'Do you already have some materials or tools available for this project? If not, no problem—I can assist with recommendations.'\n"
                "- 'What is your budget range for this project?'\n"
                "If no additional information is needed, confirm with the user that the details are complete, summarize the gathered information, "
                "and offer reassurance that the project can proceed to the next step. For example:\n"
                "- 'Great! Based on what you've shared, I have all the details I need to proceed.'\n"
                "- 'Here's a quick summary of what I understand: [summary]. Let me know if I missed anything!'"
            )
        )

    def materials_task(self, project_description):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"List the materials required for the following project: {project_description}. "
                        f"The response should only contain the MATERIALS and must NOT include the TOOLS. "
                        f"Include alternatives where applicable. "
                        f"Analyze if there is enough information to perform the task and provide an appropriate response and make sure to know all relevant details"
                        f"If information is missing, generate a specific question for the user.",
            agent=self.materials_agent(),
            expected_output=(
                "If more information from the user is required, answer 'question:' followed by a clear and specific question in the language of the user. For example:\n"
                "- 'What type of materials would you prefer for this project?'\n"
                "- 'What is the size or scope of the project to estimate the quantity of materials needed?'\n"
                "If no more information from the user is needed, answer with a markdown list of materials, including the estimated required quantities and alternatives, e.g.:\n\n"
                "- **Material 1**: High-quality cement\n"
                "  - Quantity: 10 kg\n"
                "  - Alternative: Eco-friendly cement (8 kg)\n"
                "- **Material 2**: Paint (white)\n"
                "  - Quantity: 2 liters\n"
                "  - Alternative: Matte finish paint (2 liters)\n"
            ),
            async_execution=True
    )

    def tools_task(self, project_description):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        materials = self.context['materials']
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"List the tools required for the following project: {project_description}. "
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
            ),
            async_execution=True
    )
 
    def cost_estimation_task(self, materials_list):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        materials = self.context['materials']
        tools = self.context['tools']
        location = self.context.get('user_location', 'Europe')
        currency = self.context.get('currency', '€')

        # Define reference markets based on location context or general applicability
        markets = (
            "Leroy Merlin, Castorama, OBI, Bricofer, Home Depot, Home Depot Mexico, "
            "and similar stores based on the user's location."
        )

        return Task(
            description=(
                f"Consider the conversation history: {recent_history}.\n"
                f"Provide a cost estimation for the following materials: {materials_list}.\n"
                f"Consider materials provided in context: {materials}.\n"
                f"Consider tools provided in context: {tools}.\n"
                f"Use the user's location ({location}) to determine the appropriate markets ({markets}) and currency ({currency}).\n"
                f"If the user's location is unknown, default to providing costs in euros (€).\n"
                f"Focus on direct price information (e.g., price per unit) and avoid providing unrelated context or market trends.\n"
                f"Respond in markdown table format for clarity, showing costs in the relevant currency.\n"
                f"Analyze if there is enough information to perform the task. If not, generate a specific question to ask the user for more details."
            ),
            agent=self.cost_agent(),
            expected_output=(
                "If more information from the user is required, answer 'question:' followed by a specific question in the user's language.\n"
                "If enough information is provided, respond with a markdown table of costs, referencing the relevant markets and using the appropriate currency. For example:\n\n"
                "| Material        | Cost (in {currency})  | Alternatives                       |\n"
                "|----------------|----------------------|------------------------------------|\n"
                "| Paint          | 15 €/liter          | Eco-paint (20 €/liter)             |\n"
            )
        )

    
    def guide_task(self, repair_or_renovation_process):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        materials = self.context['materials']
        tools = self.context['tools']
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"Provide detailed step-by-step instructions for the following repair or renovation process: {repair_or_renovation_process}. "
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
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        materials = self.context['materials']
        tools = self.context['tools']
        return Task(
            description=(
                f"Consider the conversation history: {recent_history}."
                f"Search for a maximum of two contractors who specialize in the following project in the specified location or nearby. "
                f"Consider materials provided in context: {materials}. "
                f"Consider tools provided in context: {tools}. "
                f"Provide contact details or links where the user can request a budget estimation. "
                f"Ensure the contractors are well-reviewed or reputable, if possible. "
                f"Analyze if there is enough information to perform the task, including location details. Make sure to know all relevant details. "
                f"If key information is missing, generate a specific question to ask the user to gather the necessary details."
            ),
            agent=self.contractor_search_agent(),
            expected_output=(
                "If more information from the user is required, answer 'question:' followed by a clear and specific question in the language of the user. For example:\n"
                "- 'Could you specify the location for the project so I can find local contractors?'\n"
                "- 'Are there any specific requirements or certifications needed for the contractors?'\n"
                "If no additional information from the user is needed, answer with a list of up to two contractors with their contact information or website links, including details on how to request a budget estimation. For example:\n\n"
                "- **Contractor 1**: ABC Renovations\n"
                "  - Contact: (123) 456-7890\n"
                "  - Website: [www.abcrenovations.com](http://www.abcrenovations.com)\n"
                "- **Contractor 2**: Home Fix Pros\n"
                "  - Contact: (987) 654-3210\n"
                "  - Website: [www.homefixpros.com](http://www.homefixpros.com)\n"
            )
        )

    def safety_task(self, task_description):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
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
    def scheduling_task(self, project_description, deadline=None):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        materials = self.context['materials']
        tools = self.context['tools']
        step_by_step_guide = self.context['step_by_step_guide']

        return Task(
            description=(
                f"Using the conversation history: {recent_history}, "
                f"create a schedule for the project: {project_description}. "
                f"Base the schedule on the following inputs:\n"
                f"- Materials provided: {materials}\n"
                f"- Tools provided: {tools}\n"
                f"- Step-by-step guide: {step_by_step_guide}\n"
                f"Provide a table with:\n"
                f"- Task\n"
                f"- Duration\n"
                f"- Recommended number of people\n"
                f"If a deadline is specified ({deadline}), prioritize tasks to meet it."
            ),
            agent=self.scheduler_agent(),
            expected_output=(
                "If more information is needed, respond with 'question:' followed by the missing details.\n"
                "If sufficient information is provided, return a detailed schedule in markdown format. Example:\n\n"
                "| Task                | Duration | Recommended People |\n"
                "|---------------------|----------|--------------------|\n"
                "| Gather Materials    | 2 days   | 3                  |\n"
                "| Prep Work Area      | 1 day    | 2                  |\n"
                "| Perform Repairs     | 3 days   | 4                  |\n\n"
                "If the project cannot be completed by the deadline, suggest adjustments."
            )
        )




    def presentation_task(self, task_description):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        materials = self.context['materials']
        tools = self.context['tools']
        step_by_step_guide = self.context['step_by_step_guide']
        contractors = self.context['contractors']
        cost_estimation = self.context['cost_estimation']
        safety_guidance = self.context['safety_guidance']
        schedule = self.context.get('schedule')  

        return Task(
            description=(
                f"Consider the conversation history: {recent_history}."
                f"Compose a final, well-structured response with the collected project details based on the user's question."
                f"Include project guidance, required materials, tools, cost estimation, step-by-step guide, recommended contractors, "
                f"safety guidance notes, and the project schedule. "
                f"Ensure that the response is visually clear, well-organized, and presented in the user's language. "
                f"If translation is necessary, adapt the response to the language detected in the user's question."
                f"Consider the question of the user: {task_description}."
                f"Consider materials in context: {materials}. (Also missing information if there is indicated)." 
                f"Consider tools in context: {tools}. (Also missing information if there is indicated)."
                f"Consider step by step guide in context: {step_by_step_guide}. (Also missing information if there is indicated)."
                f"Consider contractors in context: {contractors}. (Also missing information if there is indicated)."
                f"Consider cost estimation in context: {cost_estimation}. (Also missing information if there is indicated)."
                f"Consider safety guidance notes in context: {safety_guidance}. (Also missing information if there is indicated)."
                f"Consider the schedule in context: {schedule}. (Also missing information if there is indicated)."
            ),
            agent=self.presentation_agent(),
            expected_output=(
                "Use the language of the user and provide a structured and elegant response in markdown format, organizing all information clearly under distinct headings.\n"
                "Must be displayed in a table if this is convenient.\n"
                "Response must include all the following information:\n"
                "- Materials.\n"
                "- Tools.\n"
                "- Step by step guide.\n"
                "- Contractors with all available information (name, contact details, etc.).\n"
                "- Cost estimation.\n"
                "- Safety guidance notes.\n"
                "- Project schedule with clear timelines and dependencies.\n"
                "- Questions asking for missing information if there is any.\n"
            )
        )


    ##------------------------------------CREATE CREW------------------------------------

    def get_response(self, question):
        try:
            self.context['conversation_history'].append({"role": "user", "content": question})
            
            # Step 1: Check relevance
            relevance_task = self.check_relevance_task(question)
            relevance_crew = Crew(
                agents=[relevance_task.agent],
                tasks=[relevance_task],
                verbose=True
            )
            relevance_result = relevance_crew.kickoff()

            if relevance_result.lower().startswith('not related:'):
                return relevance_result.split(':', 1)[1].strip()

            # Step 2: Classify project type if undefined
            if not self.context['project_type']:
                classification_task = self.planificator_task(question)
                classification_crew = Crew(
                    agents=[classification_task.agent],
                    tasks=[classification_task],
                    verbose=True
                )
                project_type_result = classification_crew.kickoff()
                self.context['project_type'] = 'repair' if 'repair' in project_type_result.lower() else 'renovation'

            # Step 3: Check for questions
            questions_task = self.questions_task(question)
            questions_crew = Crew(
                agents=[questions_task.agent],
                tasks=[questions_task],
                verbose=True
            )
            questions_result = questions_crew.kickoff()

            # Branch based on whether there are questions
            if questions_result.lower().startswith('question:'):
                # If there are questions, return them and don't proceed with tasks
                return questions_result.split(':', 1)[1].strip()
            else:
                # Only proceed with task sequence if there are no questions
                task_sequence = [
                    (self.materials_task, 'materials'),
                    (self.tools_task, 'tools'),
                    (self.cost_estimation_task, 'cost_estimation'),
                    (self.guide_task, 'step_by_step_guide'),
                    (self.contractor_search_task, 'contractors'),
                    (self.safety_task, 'safety_guidance'),
                    (self.scheduling_task, 'schedule')
                ]

                # Process all tasks and store results
                task_results = {}
                for task_method, context_key in task_sequence:
                    if not self.context.get(context_key):
                        task = task_method(question)
                        crew = Crew(agents=[task.agent], tasks=[task], verbose=True)
                        result = crew.kickoff()
                        task_results[context_key] = result
                        self.context[context_key] = result

                # Final presentation only happens if we completed the task sequence
                presentation_task = self.presentation_task(question, task_results)
                presentation_crew = Crew(
                    agents=[presentation_task.agent], 
                    tasks=[presentation_task], 
                    verbose=True
                )
                final_result = presentation_crew.kickoff()

                # Reset project and update conversation history
                self.reset_project()
                self.context['conversation_history'].append({"role": "assistant", "content": final_result})

                return final_result

        except AttributeError as e:
            return f"Sorry, there was an issue with one of the tools or attributes: {str(e)}"
        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}"