# RAG LLM MANAGER
"""
    - PACKAGE class: **RAGLLM_Manager**
    - Autor: javier.sanjuan3@soprasteria.com
    - Fecha publicación: febrero 2024
"""

# Dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader, PyPDFLoader #, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    # HumanMessagePromptTemplate,
    # SystemMessagePromptTemplate
)
from time import sleep
from ftlangdetect import detect
from operator import itemgetter
import os
import yaml
import tiktoken

log_message = "[RAG LLM Manager]"

custom_rag_template_sufix = """
Use the following pieces of context to answer the question at the end.
If the question cannot be answered using information from the context, or by deducing it from the context, just say that you don't know, don't try to make up an answer.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use 4 sentences maximum but keep the answer as concise as possible.
Always say "thanks for asking!" in the specified language at the end of the answer.

Context: {context}

Question: {question}

Answer in the following language: {language}

Helpful Answer:"""

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def openai_api_calculate_cost(usage, model="gpt-4"):
    pricing = {
        'gpt-3.5-turbo': {
            'prompt': 0.0005,
            'completion': 0.0015,
        },
        'gpt-4': {
            'prompt': 0.03,
            'completion': 0.06,
        },
        'gpt-4-turbo-preview': {
            'prompt': 0.01,
            'completion': 0.03,
        }
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")

    prompt_cost = usage['prompt_tokens'] * model_pricing['prompt'] / 1000
    completion_cost = usage['completion_tokens'] * model_pricing['completion'] / 1000

    total_cost = prompt_cost + completion_cost
    # round to 6 decimals
    total_cost = round(total_cost, 6)

    return total_cost


# ===========================================================================================
# ========= CLASS RAGLLM_Manager ================================================================
# ===========================================================================================
class RAGLLM_Manager:
    # RAG LLM indexing and chatbot Manager

    # ====================== CLASS PARAMS ======================================
    _loader = None
    _splitter = None
    _retriever = None
    _prompt = None
    _contextualize_q_chain = None
    _rag_chain = None
    _chat_history = []

    # ====================== CONSTRUCTOR ==================================================
    def __init__(self, credentials_path, assistant_role_instruction):
        # Set OPENAI_API_KEY
        with open(credentials_path, "r") as stream:
            try:
                credentials = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        os.environ["OPENAI_API_KEY"] = credentials["OPENAI_API_KEY"]

        self._prompt = assistant_role_instruction + custom_rag_template_sufix

        print('{}: Objeto RAGLLM_Manager creado. ----------------------------'.format(log_message))

    # ====================== GETTERS & SETTERS ============================================
    def get_prompt(self):
        return self._prompt

    def set_prompt(self, prompt):
        self._prompt = prompt
        return self

    def get_retriever(self):
        return self._retriever

    # ====================== UTILS FUNCTIONS =========================================

    # ====================== MAIN FUNCTIONS =========================================
    def load_data_from_docx(self, directorypath):
        # Load docx documents
        self._loader = DirectoryLoader(directorypath, glob="*.docx", loader_cls=Docx2txtLoader)
        data = self._loader.load()

        print('{}: Docx files loaded from {} ----------------------------'.format(log_message, directorypath))

        return data
    # -------------------------------------------------------------------------------

    def load_data_from_pdf(self, directorypath):
        # Load pdf documents
        self._loader = DirectoryLoader(directorypath, glob="*.pdf", loader_cls=PyPDFLoader)
        data = self._loader.load()

        print('{}: PDF files loaded from {} ----------------------------'.format(log_message, directorypath))

        return data
    # -------------------------------------------------------------------------------

    def split_text_data(self, data, chunk_size=1000, chunk_overlap=200):
        # Split documents into chunks of text data
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = self._splitter.split_documents(documents=data)

        print('{}: Data splitted ----------------------------'.format(log_message))

        return splits
    # -------------------------------------------------------------------------------

    def index_faiss(self, splits):
        # Indexing of the chunks of text data
        vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        self._retriever = vectorstore.as_retriever()

        print('{}: Data indexed. Retriever generated. ----------------------------'.format(log_message))
    # -------------------------------------------------------------------------------

    def create_chatbot(self, model="gpt-3.5-turbo", temperature=0):
        llm = ChatOpenAI(model_name=model, temperature=temperature)
        custom_rag_prompt = PromptTemplate.from_template(self._prompt)

        self._rag_chain = (
            {
                "context": itemgetter("question") | self._retriever,
                "question": itemgetter("question"),
                "language": itemgetter("language"),
            }
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )

    # -------------------------------------------------------------------------------
    def contextualized_question(self, input: dict):
        print(f"chat_history?: {input.get('chat_history')}")
        if input.get("chat_history"):
            return self._contextualize_q_chain
        else:
            return input["question"]

    def create_chatbot_with_memory(self, model="gpt-3.5-turbo", temperature=0):
        llm = ChatOpenAI(model_name=model, temperature=temperature)

        # Chain for chat history
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self._contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

        # custom_rag_prompt = PromptTemplate.from_template(self._prompt)

        custom_rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        self._rag_chain = (
            RunnablePassthrough.assign(
                context=self.contextualized_question | self._retriever | format_docs
            )
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )

    # -------------------------------------------------------------------------------
    def retrieve_and_print(self, question):
        retrieved_docs = self._retriever.invoke(question)

        for index, retrieved_doc in enumerate(retrieved_docs):
            print("\r\n-------------------------------------------------------------------------------------")
            print(f"Fragmento de documento: {index}")
            print(retrieved_doc.page_content)

    # -------------------------------------------------------------------------------
    def generate_and_stream(self, question):
        for chunk in self._rag_chain.stream({"question": question, "language": detect(text=question, low_memory=False)["lang"]}):
            print(chunk, end="", flush=True)
            sleep(0.05)

    # -------------------------------------------------------------------------------
    def generate(self, question):
        idioma_detectado = detect(text=question, low_memory=False)["lang"]
        print("-------------------------------------------------------------------------------------")
        print(f"Idioma detectado: {idioma_detectado}")
        return self._rag_chain.invoke({"question": question, "language": idioma_detectado})

    # -------------------------------------------------------------------------------
    def generate_with_history_update(self, question, model):
        idioma_detectado = detect(text=question, low_memory=False)["lang"]
        print("-------------------------------------------------------------------------------------")
        print(f"Idioma detectado: {idioma_detectado}")
        ai_msg = self._rag_chain.invoke({"chat_history": self._chat_history, "question": question, "language": idioma_detectado})
        self._chat_history.extend([HumanMessage(content=question), ai_msg])
        print("-------------------------------------------------------------------------------------")
        print(f"Chat History: {self._chat_history}")
        print("-------------------------------------------------------------------------------------")
        print("PRICING")
        string_chat_history = ' '.join([message for message in self._chat_history if isinstance(message, str)])
        self.calculate_cost(question+' '+string_chat_history, ai_msg, model)
        return ai_msg

    # -------------------------------------------------------------------------------
    def calculate_cost(self, prompt_str, completion_str, model):
        # Number of tokens calculation
        usage = {'prompt_tokens': num_tokens_from_string(prompt_str, model),
                 'completion_tokens': num_tokens_from_string(completion_str, model)}

        total_cost = openai_api_calculate_cost(usage, model)
        print(f"\nTokens used:  {usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} completion = {usage['prompt_tokens'] + usage['completion_tokens']:,} tokens")
        print(f"Total cost: ${total_cost:.4f}\n")

        return total_cost
