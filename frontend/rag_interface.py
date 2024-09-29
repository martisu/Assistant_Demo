import streamlit as st
import time
from src.ragllm_manager import RAGLLM_Manager
import yaml
import os

from PIL import Image

config_path = "./config/"

st.set_page_config(
   page_title="Sopra Steria - Assets",
   page_icon="🧊",
   layout="wide",
   # initial_sidebar_state="expanded"
)

# ============ FUNCTIONS ==========================================================================
# RAG creation
def rag_creation(directorypath, chunk_size, chunk_overlap, model, temperature, credentials_path, assistant_role_instruction):
    ragllm = RAGLLM_Manager(credentials_path, assistant_role_instruction)

    data = []

    for fname in os.listdir(directorypath):
        if fname.endswith('.docx') or fname.endswith('.pdf'):
            break
    else:
        print(f"No hay informes de estudiantes .docx or .pdf en el directorio: {directorypath} - Por favor, incluya documentos de contexto.")
        return None

    # Reading docx files
    data = ragllm.load_data_from_docx(directorypath)
    # Reading pdf files
    data = data + ragllm.load_data_from_pdf(directorypath)

    # Split data
    splits = ragllm.split_text_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Index data
    ragllm.index_faiss(splits)

    # Creating RAGLLM chatbot
    # ragllm.create_chatbot(model=model, temperature=0.2)
    ragllm.create_chatbot_with_memory(model=model, temperature=temperature)
    return ragllm


# Streamed response emulator
def response_generator(question, ragllm, model):
    print("=====================================================================================")
    print(f"Question: {question}")
    ragllm.retrieve_and_print(question)
    response = ragllm.generate_with_history_update(question, model)
    print("-------------------------------------------------------------------------------------")
    print(f"Response: {response}")

    for line in response.split('\n'):
        yield '\r\n'
        for word in line.split():
            yield word + " "
            time.sleep(0.05)


# ============ INTERFACE ==========================================================================
image = Image.open('./frontend/img/logo-soprasteria.png')
st.image(image, caption='')
#st.sidebar.title("Demostrador Sopra 📊")
#st.sidebar.lis

st.title("Consulta de Informes de Estudiantes")

# Initialize RAGLLM model
if "RAGLLM_model" not in st.session_state:
    print("Loading RAGLLM model")
    # Reading params
    with open(config_path+"ragllm_params.yml", "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print(params["datapath"])
    # Creating RAGLLM module
    st.ragllm = rag_creation(directorypath = params["datapath"],
                             chunk_size = params["chunk_size"],
                             chunk_overlap = params["chunk_overlap"],
                             model = params["model"],
                             temperature = params["temperature"],
                             credentials_path = params["credentials_path"],
                             assistant_role_instruction = params["assistant_role_instruction"])

    if st.ragllm is None:
        st.stop()

    st.session_state["RAGLLM_model"] = params["model"]
    print("RAGLLM model loaded")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input

if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, st.ragllm, st.session_state["RAGLLM_model"]))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})