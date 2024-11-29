
from crewai import Agent, Task, Crew, Process
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from threading import Event
from threading import Thread


from crewai_tools import ScrapeWebsiteTool
import yaml
import os
import time

def retry_with_backoff(func, max_retries=30):
    def wrapper(*args, **kwargs):
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if '429' in str(e) and i < max_retries - 1:
#                    wait_tmp = random.randint(1, 10)
                    wait_tmp = i + random.uniform(1, 5)
                    print('==================', wait_tmp)
                    time.sleep(wait_tmp)
                    continue
                raise
    return wrapper

class CrewAIChatbot:

    HISTORY_LIMIT = 50  # Length of the history to consider

    def __init__(self, credentials_path):
        self.credentials = self.load_credentials(credentials_path)

        # Initialize task events
        self.task_events = {
#            'schedule': Event(),
            'materials': Event(),
            'tools': Event(),
            'contractors': Event(),
            'safety_guidance': Event(),
            'cost_estimation': Event()
        }

        self.task_dependencies = {
            'materials': set(),  
            'tools': set(),      
            'contractors': set(),
            'safety_guidance': set(),
            'cost_estimation': {'materials', 'tools'} 
        }


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
            temperature=0.1,
#            max_tokens = 250

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
            'schedule': None,
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
            'schedule': None,
            'step_by_step_guide': None,
            'contractors': [],
            'safety_guidance': None,
            'conversation_history': conversation_history,
        }
        # Reset all events
        for event in self.task_events.values():
            event.clear()



    def execute_task(self, task_method, context_key, question, execution_times):
            """Execute a single task in a separate thread"""
            start_time = time.time()
            task = task_method(question)
            crew = Crew(agents=[task.agent], tasks=[task], verbose=True)
            result = crew.kickoff()
                
            if not result.lower().startswith('question:'):
                self.context[context_key] = result
                self.task_events[context_key].set()
                print(f"Task {context_key} completed and event set")
                    
            execution_times[context_key] = round(time.time() - start_time, 2)
            print(f"{context_key.replace('_', ' ').title()} took: {execution_times[context_key]} seconds")


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
            role='Relevance Checker and Information Gatherer',
            goal='Determine if a query is related to home improvement projects and gather all necessary information.',
            tools=[],
            verbose=True,
            backstory=(
                "You are an expert in home improvement projects with excellent communication skills. "
                "Your task has TWO parts:\n"
                "1. Determine if a question is about home repairs, renovations, or any other home improvement task.\n"
                "2. If it is related, analyze what information would be needed for ALL phases of the project:\n"
                "   - Dimensions and specifications\n"
                "   - Material preferences and quantities\n"
                "   - Tools and equipment needs\n"
                "   - Budget constraints\n"
                "   - Timeline requirements\n"
                "   - Location details\n"
                "   - Safety considerations\n"
                "Always respond in the language of the user and ensure all gathered information is complete before proceeding. "
                "Ensure all essential user-specific information is obtained."
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
                "You are a VERY quick and efficient construction cost estimator. "
                "Your role is to provide cost estimations for materials and tools"
#                "Default to euros (€) if the user’s location is not specified. "
#                "Perform a targeted search for pricing data and ensure clarity in the response. "
                "Provide approximate unit prices in the user’s currency or a specified currency. "
                "Avoid including unrelated context or general market trends."
            ),
            llm=self.llm
        )
    
#    def guide_agent(self):
#        return Agent(
##            role='Step-by-Step Guide',
#            goal='Provide detailed step-by-step instructions for any task.',
#            tools=[self.search_tool] + self.pdf_tools,
#            verbose=True,
#            backstory=(
#                "You are an expert guide. Your role is to break down complex tasks into clear, manageable steps."
#                "Always ensure the instructions are simple and precise, adjusting based on the user's feedback."
#                "Provide explanations for each step, but keep them concise."
#                "Always detect the language of the user's input and respond in that language unless explicitly instructed otherwise." 
#                "Respond using the currency of the user’s location if specified; otherwise, default to euros."
#            ),
#            llm=self.llm
#        )
    
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
                "adapt instructions based on the context and task complexity."
            ),
            llm=self.llm
        )
    
    def scheduler_agent(self):
        return Agent(
            role='Project Analyzer and Scheduler',
            goal='Analyze project requirements, identify missing information, and create schedule with guidance.',
            tools=[self.search_tool] + self.pdf_tools,
            verbose=True,
            backstory=(
                "You are a quick and eficient project analysis expert for work planning. "
                "You create a detailed schedule and step-by-step guide. "
                "You think systematically about ALL aspects that other specialists would need to know: "
                "dimensions, materials preferences, budget constraints, timeline requirements, location details, "
                "specific requirements for contractors, safety considerations, etc."
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
                "You are VERY quick and efficient presentation expert responsible for presenting project information "
                "in a clear and structured format. Your role is to create a coherent response that includes project guidance, "
                "required materials, tools, cost estimation, step-by-step guide, and recommended contractors. "
                "Ensure the response is understandable, visually organized, and presented in the user's language."
            ),
            llm=self.llm
        )


##------------------------------------TASKS------------------------------------
    
    @retry_with_backoff
    def check_relevance_task(self, question):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]

        return Task(
            description=(
                f"Analyze the user's query: {question} in the conversation history:\n{recent_history}\n\n"
                f"Follow these steps internally (but DO NOT include these steps in your response):\n"
                f"1. Check if query is related to home improvement\n"
                f"2. If related, analyze what information is needed\n\n"
                f"3. Review conversation history to avoid repetitive phrasing\n"
                f"Then, RESPOND USING EXACTLY ONE of these formats:\n"
                f"- If not related: Start with 'NOT RELATED: ' followed by friendly redirection\n"
                f"- If related but missing info: Start with 'QUESTION: ' followed by friendly most important brief missing info summary + ONE question\n"
                f"- If related and complete: Start with 'RELATED: ' followed by confirmation\n\n"
                f"IMPORTANT:\n"
                f"- DO NOT include phase descriptions or any other prefixes in your response\n"
                f"- Respond in the same language as the user's query\n"
                f"- If information is missing, first list what's missing briefly, then ask ONE key question.\n"
#                f"- Only ask about user preferences, location, or project requirements and conditions.\n"
                f"- Do NOT ask about technical decisions that your expertise should make (such as tools required).\n"
#                f"- Ensure all essential user-specific information is obtained, such as (preferences, location, requerimetns, codnconditions, etc ).\n"
                f"- This essential information must not be missing: Preferences, location, requerimetns, codnconditions.\n"
                f"- Vary your phrasing based on previous responses in the conversation history\n"
                f"- Start DIRECTLY with one of the three specified prefixes"
            ),
            agent=self.relevance_agent(),
            expected_output=(
                "MUST return EXACTLY ONE of these three formats:\n"
                "1. 'NOT RELATED: ' followed by friendly redirection in the language of the user\n"
                "2. 'QUESTION: '  ' followed by brief missing info summary + ONE question\n"
                "3. 'RELATED'\n"
            )
        )

    @retry_with_backoff
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
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        project_type = self.context["project_type"]

        # Format recent history for inclusion in the task
        history_str = "\n".join([f"{entry['role']}: {entry['content']}" for entry in recent_history])

        # Define the task
        return Task(
            description=(
                f"Considering the following recent conversation history:\n\n{history_str}\n\n"
                f"Analyze the user's query: '{question}' in the context of the conversation and the project type: '{project_type}'.\n\n"
                f"Determine if additional questions are needed to gather missing details about the project.\n"
                f"If further clarification is required, respond with 'Question: ' followed by specific, clear, and professional questions tailored to the context. "
                f"Examples include:\n"
                f"- 'What are the dimensions of the area to be worked on?'\n"
                f"- 'Do you already have some materials or tools available for this project?'\n"
                f"- 'What is your budget range for this project?'\n\n"
                f"If no further questions are needed, respond with 'Complete: ' followed by a confirmation message and a summary of the gathered information. "
                f"Examples include:\n"
                f"- 'Great! Based on what you've shared, I have all the details I need to proceed.'\n"
                f"- 'Here's a quick summary of what I understand: [summary]. Let me know if I missed anything!'"
            ),
            agent=self.questions_agent(),
            expected_output=(
                "A response prefixed with either:\n"
                "- 'Question: ' followed by specific questions to gather missing details.\n"
                "- 'Complete: ' followed by a confirmation and summary of gathered details."
            )
        )

    @retry_with_backoff
    def materials_task(self, project_description):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        schedule = self.context['schedule']
        print('schedule::::::::::::::::::::::::', schedule)
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"Consider schedule provided in context: {schedule}. "
                        f"List the materials required for the following project: {project_description}. "
                        f"The response should only contain the MATERIALS and must NOT include the TOOLS. "
                        f"Include alternatives where applicable. ",
            agent=self.materials_agent(),
            expected_output=(
                " Response using the minimum characters needed\n"
                " Answer with a markdown list of materials, including the estimated required quantities and alternatives, e.g.:\n\n"
                "- **Material 1**: High-quality cement\n"
                "  - Quantity: 10 kg\n"
                "  - Alternative: Eco-friendly cement (8 kg)\n"
                "- **Material 2**: Paint (white)\n"
                "  - Quantity: 2 liters\n"
                "  - Alternative: Matte finish paint (2 liters)\n"
            ),
            async_execution=True
    )

    @retry_with_backoff
    def tools_task(self, project_description):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        schedule = self.context['schedule']
        return Task(
            description=f"Consider the conversation history: {recent_history}."
                        f"List the tools required for the following project: {project_description}. "
                        f"Consider schedule provided in context: {schedule}. "
                        f"The response should only contain the TOOLS and their quantities, without including MATERIALS. ",
            agent=self.tools_agent(),
            expected_output=(
                " Response using the minimum characters needed\n"
                " Answer with a markdown list of tools and their alternatives, e.g.:\n\n"
                "- **Tool 1**: Electric drill\n"
                "  - Alternative: Manual drill\n"
                "- **Tool 2**: Screwdriver (Phillips)\n"
                "  - Alternative: Flathead screwdriver\n"
            ),
            async_execution=True
    )
 
    @retry_with_backoff
    def cost_estimation_task(self, project_description):

        materials_ready = self.task_events['materials'].wait(timeout=300)
        tools_ready = self.task_events['tools'].wait(timeout=300)

        if not (materials_ready and tools_ready):
            materials = 'Unknown materials'
            tools = 'Unknown tools'
            print('\nTimeOut Unknown aterials and/or tools for cost calculation.\n')            
        else:
            materials = self.context['materials']
            tools = self.context['tools']
            print('MATERIALS::::::::::::::::::::::::', materials)
            print('TOOLS::::::::::::::::::::::::::::', tools)

        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
#        schedule = self.context['schedule']
#        location = self.context.get('user_location', 'Europe')
#        currency = self.context.get('currency', '€')

        # Define reference markets based on location context or general applicability
        markets = (
            "Leroy Merlin, Castorama, OBI, Bricofer, Home Depot, Home Depot Mexico, "
            "and similar stores based on the user's location."
        )

        return Task(
            description=(
                f"Provide a quick cost estimation for the required petition: {project_description}.\n"
                f"Consider the conversation history context: {recent_history}.\n"
#                f"Consider schedule provided in context: {schedule}.\n"
                f"Consider materials provided in context: {materials}.\n"
                f"Consider tools provided in context: {tools}.\n"
#                f"Use the user's location ({location}) to determine the appropriate markets ({markets}) and currency ({currency}).\n"
                f"Consider the required location.\n"
#                f"Focus on direct price information (e.g., price per unit) and avoid providing unrelated context or market trends.\n"
#                f"Respond in markdown table format for clarity, showing costs in the relevant currency.\n"
                f"Respond using the number format (US: 1,234.56 or EU: 1.234,56) and measurement system (metric/imperial) specified by the user. "
                 "If none specified, use their location's standard. Default to European format (EU numbers, metric) if no location given."
            ),
            agent=self.cost_agent(),
            expected_output=(
                "Response using the minimum characters needed\n"
                "Respond with a markdown table of costs, referencing the relevant markets and using the appropriate currency. For example:\n\n"
                "| Material        | Cost (in €)          | Alternatives                       |\n"
                "|-----------------|----------------------|------------------------------------|\n"
                "| Paint           | 15 €/liter           | Eco-paint (20 €/liter)             |\n"
            ),
#            async_execution=True
        )

#    @retry_with_backoff
#    def guide_task(self, repair_or_renovation_process):
#        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
#        materials = self.context['materials']
#        tools = self.context['tools']
#        return Task(
#            description=f"Consider the conversation history: {recent_history}."
#                        f"Provide detailed step-by-step instructions for the following repair or renovation process: {repair_or_renovation_process}. "
#                        f"Consider materials provided in context: {materials}. "
#                        f"Consider tools provided in context: {tools}. "
#                        f"Ensure that the steps are easy to follow and comprehensive, covering all necessary tools, materials, and safety precautions. ",
#            agent=self.guide_agent(),
#            expected_output=(
#                "Answer with a list of detailed steps for the repair or renovation process. For example:\n\n"
#                "1. Identify the scope of the repair or renovation.\n"
#                "2. Gather all necessary tools and materials.\n"
#                "3. Prepare the work area to ensure safety and efficiency.\n"
##                "4. Step-by-step breakdown of the actual work (e.g., removing old materials, installing new ones).\n"
#                "5. Final touches and clean-up instructions.\n"
#            ),
#            async_execution=True
#        )
    
    @retry_with_backoff
    def contractor_search_task(self, project_description):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        materials = self.context['materials']
        tools = self.context['tools']
        schedule = self.context['schedule']
        return Task(
            description=(
                f"Consider the conversation history: {recent_history}."
                f"Search for a maximum of two contractors who specialize in the following project in the specified location or nearby. "
                f"Consider materials provided in context: {schedule}. "
                f"Consider materials provided in context: {materials}. "
                f"Consider tools provided in context: {tools}. "
                f"Provide contact details or links where the user can request a budget estimation. "
                f"Ensure the contractors are well-reviewed or reputable, if possible. "
            ),
            agent=self.contractor_search_agent(),
            expected_output=(
                "Response using the minimum characters needed.\n"
                "Answer with a list of up to two contractors with their contact information or website links, including details on how to request"
                " a budget estimation. For example:\n\n"
                "- **Contractor 1**: ABC Renovations\n"
                "  - Contact: (123) 456-7890\n"
                "  - Website: [www.abcrenovations.com](http://www.abcrenovations.com)\n"
                "- **Contractor 2**: Home Fix Pros\n"
                "  - Contact: (987) 654-3210\n"
                "  - Website: [www.homefixpros.com](http://www.homefixpros.com)\n"
            ),
            async_execution=True
        )

    @retry_with_backoff
    def safety_task(self, task_description):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        schedule = self.context['schedule']
        materials = self.context['materials']
        tools = self.context['tools']
#        step_by_step_guide = self.context['step_by_step_guide']
        return Task(
            description=(
                f"Consider the conversation history: {recent_history}."
                f"Provide a careful, safety-focused guide for the following task: {task_description}. "
                f"The instructions should prioritize accident prevention by outlining each step in detail, highlighting any safety risks,"
                f"and suggesting appropriate protective measures or precautions. Emphasize where extra caution is needed."
                f"Consider the question of the user: {task_description}."
                f"Consider materials in context: {materials}. " 
                f"Consider schedule in context: {schedule}. " 
                f"Consider tools in context: {tools}. "
#                f"Consider step by step guide in context: {step_by_step_guide}. " 
            ),
            agent=self.safety_agent(),
            expected_output=(
                "Response using the minimum characters needed. \n"
                "Respond with a concise step-by-step safety guide. Include the following considerations:\n"
                "- Key steps to perform the task safely."
                "- Specific risks associated with each step."
                "- Protective measures to mitigate risks."
                "- Areas where extra caution is needed."
            ),
            async_execution=True
        )

    @retry_with_backoff
    def scheduling_task(self, project_description, deadline=None):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        return Task(
            description=(
                f"Using the conversation history: {recent_history}, create both a schedule and step-by-step guide for the project: {project_description}. "
                f"The response should include TWO clearly separated sections:\n"
                f"1. SCHEDULE: A table with Task, Duration, and Recommended number of people\n"
                f"2. STEP-BY-STEP GUIDE: Detailed instructions for each task\n"
                f"If a deadline is specified ({deadline}), prioritize tasks to meet it.\n"
                f"Ensure all safety considerations and prerequisites are included in both sections."
            ),
            agent=self.scheduler_agent(),
            expected_output=(
                "Return only schedule and guide in markdown format:\n\n"
                "## Schedule\n"
                "| Task                | Duration | Recommended People |\n"
                "|---------------------|----------|--------------------|\n"
                "| Gather Materials    | 2 days   | 3                  |\n"
                "| Prep Work Area      | 1 day    | 2                  |\n\n"
                "## Step-by-Step Guide\n"
                "1. Preparation Phase:\n"
                "   - Gather all materials\n"
                "   - Set up work area\n"
                "2. Execution Phase:\n"
                "   - Detailed steps for execution\n"
                "3. Final Phase:\n"
                "   - Clean-up and inspection steps\n"
            )
        )

    @retry_with_backoff
    def presentation_task(self, task_description):
        recent_history = self.context['conversation_history'][-self.HISTORY_LIMIT:]
        materials = self.context['materials']
        tools = self.context['tools']
#        step_by_step_guide = self.context['step_by_step_guide']
        contractors = self.context['contractors']
        cost_estimation = self.context['cost_estimation']
        safety_guidance = self.context['safety_guidance']
        schedule = self.context['schedule']
#        schedule = self.context.get('schedule')  

        return Task(
            description=(
                f"Compose a final, well-structured response with the collected project details based on rquirements."
                f"Consider the conversation history: {recent_history}."
 #               f"Include project guidance, required materials, tools, cost estimation, step-by-step guide, recommended contractors, "
#                f"safety guidance notes, and the project schedule. "
                f"Ensure that the response is visually clear, well-organized, and presented in the user's language. "
                f"If translation is necessary, adapt the response to the language detected in the user's question."
                f"Consider the question of the user: {task_description}."
                f"Consider materials in context: {materials}. (Also missing information if there is indicated)." 
                f"Consider tools in context: {tools}. (Also missing information if there is indicated)."
#                f"Consider step by step guide in context: {step_by_step_guide}. (Also missing information if there is indicated)."
                f"Consider contractors in context: {contractors}. (Also missing information if there is indicated)."
                f"Consider cost estimation in context: {cost_estimation}. (Also missing information if there is indicated)."
                f"Consider safety guidance notes in context: {safety_guidance}. (Also missing information if there is indicated)."
                f"Consider the schedule in context: {schedule}. (Also missing information if there is indicated)."
                f"Respond using the number format (US: 1,234.56 or EU: 1.234,56) and measurement system (metric/imperial) specified by the user. "
                 "If none specified, use their location's standard. Default to European format (EU numbers, metric) if no location given."
            ),
            agent=self.presentation_agent(),
            expected_output=(
                "Use the language of the user.\n"
                "Response using the minimum characters needed.\n"
                "Provide a structured and elegant response in markdown format, organizing all information clearly under distinct headings.\n"
                "Must be displayed in a table if this is convenient.\n"
                "Response must include all the following information:\n"
                "- Materials (all known information including quantity).\n"
                "- Tools (all known information including quantity).\n"
                "- Step by step guide.\n"
                "- Contractors with all available information (name, contact details, etc.).\n"
                "- Cost estimation.\n"
                "- Safety guidance notes.\n"
                "- Project schedule with clear timelines and dependencies.\n"
#                "- Questions asking for missing information if there is any.\n"
            )
        )


    ##------------------------------------CREATE CREW------------------------------------

    def get_response(self, question):
        try:

            self.context['conversation_history'].append({"role": "user", "content": question})
            execution_times = {} # Test code - Kevin - Ernest

            # Reset all events at the start
            for event in self.task_events.values():
                event.clear()


            # Step 0: Check relevance
            start_time = time.time() # Test code - Kevin - Ernest
            relevance_task = self.check_relevance_task(question)
            relevance_crew = Crew(
                agents=[relevance_task.agent],
                tasks=[relevance_task],
                verbose=True
            )
            relevance_result = relevance_crew.kickoff()
            execution_times['relevance'] = round(time.time() - start_time, 2) # Test code - Kevin - Ernest
            print(f"Relevance check took: {execution_times['relevance']} seconds") # Test code - Kevin - Ernest

            if relevance_result.lower().startswith('not related:') or relevance_result.lower().startswith('question:'):
                return relevance_result.split(':', 1)[1].strip()

            # Step 1: Classify project type if undefined
            if not self.context['project_type']:
                start_time = time.time()  # Test code - Kevin - Ernest
                classification_task = self.planificator_task(question)
                classification_crew = Crew(
                    agents=[classification_task.agent],
                    tasks=[classification_task],
                    verbose=True
                )
                project_type_result = classification_crew.kickoff()
                self.context['project_type'] = 'repair' if 'repair' in project_type_result.lower() else 'renovation'

                execution_times['classification'] = round(time.time() - start_time, 2)  # Test code - Kevin - Ernest
                print(f"Classification took: {execution_times['classification']} seconds")  # Test code - Kevin - Ernest


            if not self.context['schedule']:
                start_time = time.time()  # Test code - Kevin - Ernest
                scheduling_task = self.scheduling_task(question)
                scheduling_crew = Crew(
                    agents=[scheduling_task.agent],
                    tasks=[scheduling_task],
                    verbose=True
                )
                schedule_result = scheduling_crew.kickoff()

                # Take results
                self.context['schedule'] = schedule_result
#                self.task_events['schedule'].set()


            # Group tasks by dependency level
            task_groups = {
#                'level1': [(self.scheduling_task, 'schedule')],  # No dependencies
                'level2': [  # Depends on schedule
                    (self.materials_task, 'materials'),
                    (self.tools_task, 'tools'),
                    (self.contractor_search_task, 'contractors'),
                    (self.safety_task, 'safety_guidance')
                ],
                'level3': [  # Depends on materials and tools
                    (self.cost_estimation_task, 'cost_estimation')
                ]
            }

#            # Sequentially process tasks
#            sequential_tasks = [
#                (self.scheduling_task, 'schedule'),
#                (self.materials_task, 'materials'),
#                (self.tools_task, 'tools'),
##                (self.guide_task, 'step_by_step_guide'),
#                (self.contractor_search_task, 'contractors'),
#                (self.safety_task, 'safety_guidance'),
#                (self.cost_estimation_task, 'cost_estimation'),
#            ]


#            for task_method, context_key in sequential_tasks:
#                if not self.context.get(context_key):

#                    start_time = time.time() # Test code - Kevin - Ernest
#                    task = task_method(question)
#                    crew = Crew(agents=[task.agent], tasks=[task], verbose=True)
#                    result = crew.kickoff()
#
#                    if result.lower().startswith('question:'):
#                        return result.split(':', 1)[1].strip()
#                    
#                    self.context[context_key] = result
#                    self.task_events[context_key].set()  # Signal task completion
#                    execution_times[context_key] = round(time.time() - start_time, 2) # Test code - Kevin - Ernest
#                    print(f"{context_key.replace('_', ' ').title()} took: {execution_times[context_key]} seconds") # Test code - Kevin - Ernest

            # In get_response method:
            for level, tasks in task_groups.items():
                # Create threads for all tasks in this level
                threads = []
                for task_method, context_key in tasks:
                    if not self.context.get(context_key):
                        # Check if dependencies are met
                        dependencies = self.task_dependencies.get(context_key, set())
                        if dependencies:
                            for dep in dependencies:
                                if not self.task_events[dep].is_set():
                                    print(f"Waiting for {dep} to complete before starting {context_key}")
                                    self.task_events[dep].wait(timeout=300)
                        
                        # Start task in new thread
                        thread = Thread(
                            target=self.execute_task,
                            args=(task_method, context_key, question, execution_times)
                        )
                        threads.append(thread)
                        thread.start()
                
                # Wait for all tasks in this level to complete
                for thread in threads:
                    thread.join()




            # Step 8: Presentation
            start_time = time.time() # Test code - Kevin - Ernest
            presentation_task = self.presentation_task(question)
            presentation_crew = Crew(agents=[presentation_task.agent], tasks=[presentation_task], verbose=True)
            final_result = presentation_crew.kickoff()
            execution_times['presentation'] = round(time.time() - start_time, 2) # Test code - Kevin - Ernest
            print(f"Presentation took: {execution_times['presentation']} seconds") # Test code - Kevin - Ernest

            # Print total execution time # Test code - Kevin - Ernest
            total_time = round(sum(execution_times.values()), 2) # Test code - Kevin - Ernest
            print(f"\nTotal execution time: {total_time} seconds") # Test code - Kevin - Ernest

            # Print execution time summary # Test code - Kevin - Ernest
            print("\nExecution time summary:") # Test code - Kevin - Ernest
            for task, time_taken in execution_times.items(): # Test code - Kevin - Ernest
                percentage = round((time_taken / total_time) * 100, 1) # Test code - Kevin - Ernest
                print(f"{task.replace('_', ' ').title()}: {time_taken}s ({percentage}%)") # Test code - Kevin - Ernest

            # Reset project after response
            self.reset_project()

            self.context['conversation_history'].append({"role": "assistant", "content": final_result})

            return final_result

        except AttributeError as e:
            return f"Sorry, there was an issue with one of the tools or attributes: {str(e)}"
        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}"
