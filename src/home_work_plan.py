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
                "Always use the language the user uses unless they tell you otherwise."
                "Use 4 sentences maximum but keep the answer as concise as possible."
                "Always say 'other question' in the specified language at the end of the answer."
            ),
            llm=self.llm
        )
    
    def notes(self):
        return Agent(
            role='Project Classifier',
            goal='Classify discret information from text',
            tools=[search_tool],
            verbose=True,
            backstory=(
                "You are an expert in classifying home improvement projects. "
                "Your role is to determine whether a project is."
                "Answare in one word."
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
                "Always use the language the user uses unless they tell you otherwise."
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
                "Always use the language the user uses unless they tell you otherwise."
                "Use 4 sentences maximum but keep the answer as concise as possible."
                "Always say 'other question' in the specified language at the end of the answer."
            ),
            llm=self.llm
        )
    
    def domain_classifier(self):
        return Agent(
            role='Domain Classifier',
            goal='Determine if the question is about renovation, improvement or repear at home or another topic.',
            verbose=True,
            backstory=(
                "You are responsible for classifying whether a question is related to home improvement projects "
                "such as repairs or renovations, or if it belongs to a different topic altogether. "
                "Focus in the last line and its meaning considering previouse lines"
                "Return only the words 'home improvement' or 'other topic' without any additional text."
            ),
            llm=self.llm
        )

    # TASKS
    def classify_project_task(self, question):
        agent = self.notes()
        return Task(
            description=f"Classify the following home improvement project: {question}",
            agent=agent,
            expected_output="Classification of the project as either 'repair', 'renovation' or 'undefined'."
        )

    def provide_guidance_task(self, question, project_type):
        agent = self.repair_expert() if project_type == 'repair' else self.renovation_expert()
        return Task(
            description=f"Provide detailed guidance for this {project_type} project: {question}",
            agent=agent,
            expected_output=(
                "A comprehensive guide for the home improvement project, including step-by-step instructions, "
                "required materials, estimated time and cost, and any safety precautions."
                "Use the language of the user."
            )
        )

    def classify_domain_task(self, question):
        agent = self.domain_classifier()
        return Task(
            description=f"Classify the following question: {question}",
            agent=agent,
            expected_output="Classification of the question as either 'home improvement' or 'other topic'."
        )

    # Generate conversation context
    def generate_conversation_context(self, conversation_history):
        return "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in conversation_history])


    # Main response function
    def get_response(self, question, conversation_history):
        try:
            # Generate context based on conversation history
            context = self.generate_conversation_context(conversation_history)
            full_input = f"Context:\n{context}\n\nUser question: {question}"

            # First, classify if the question is related to home improvement
            domain_task = self.classify_domain_task(full_input)
            domain_crew = Crew(
                agents=[domain_task.agent],
                tasks=[domain_task],
                verbose=2
            )
            domain = domain_crew.kickoff()

            if "home improvement" in domain.lower():
                if 'project_type' not in locals():
                    project_type = 'undefined'
                if project_type == 'undefined':
                    # Proceed with classification and guidance as planned
                    classification_task = self.classify_project_task(question)
                    classification_crew = Crew(
                        agents=[classification_task.agent],
                        tasks=[classification_task],
                        verbose=2
                    )
                    project_type = classification_crew.kickoff()

                if project_type == 'undefined':
                    return "Podrias aclarar si se trata de un proyecto de reparación o de mejora?"

                guidance_task = self.provide_guidance_task(question, project_type)
                guidance_crew = Crew(
                    agents=[guidance_task.agent],
                    tasks=[guidance_task],
                    verbose=2
                )
                result = guidance_crew.kickoff()

                # Add the conversation to the history (handled outside this class)
                return result

            else:
                # If it's not related to home improvement, return a fallback response
                return "Estoy aquí para dar soporte a proyectos de obras en casa. Enfoquemos la conversación en este sentido."

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return f"Lo siento, he encontrado un error: {str(e)}"
