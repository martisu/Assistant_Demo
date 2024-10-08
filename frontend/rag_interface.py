import time
import yaml
import os
import streamlit as st
from PIL import Image
from src.home_work_plan import CrewAIChatbot

config_path = "./config/"

st.set_page_config(
   page_title="Home Improvement Assistant",
   page_icon="🏡",
   layout="wide",
)

# Initialize CrewAI chatbot
if "crewai_chatbot" not in st.session_state:
    credentials_path = config_path + "credentials.yml"
    st.session_state.crewai_chatbot = CrewAIChatbot(credentials_path)

# Load and display logo
image = Image.open('./frontend/img/logo-soprasteria.png')
st.image(image, caption='')

st.title("Home Improvement Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Que construimos hoy?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from CrewAI chatbot
    with st.chat_message("assistant"):
        response = st.session_state.crewai_chatbot.get_response(prompt)
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})