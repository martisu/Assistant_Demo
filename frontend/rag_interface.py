# app.py

import streamlit as st
from PIL import Image
from src.home_work_plan import CrewAIChatbot

config_path = "./config/"

st.set_page_config(
   page_title="Asistente de Mejoras del Hogar",
   page_icon="🏡",
   layout="wide",
)

# Inicializar CrewAI chatbot
if "crewai_chatbot" not in st.session_state:
    credentials_path = config_path + "credentials.yml"
    st.session_state.crewai_chatbot = CrewAIChatbot(credentials_path)

# Cargar y mostrar logo
logo_image = Image.open('./frontend/img/logo-soprasteria.png')
st.image(logo_image, caption='')

st.title("Asistente de Mejoras del Hogar")

# Image input for users
uploaded_image = st.file_uploader("Sube una imagen del área a mejorar", type=["jpg", "jpeg", "png"])
if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen cargada por el usuario")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes del historial de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Aceptar entrada del usuario
if prompt := st.chat_input("¿Qué construimos hoy?"):
    # Añadir mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)

    # Obtener respuesta de CrewAI
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = st.session_state.crewai_chatbot.get_response(prompt)
            st.markdown(response)
    
    # Añadir respuesta del asistente al historial
    st.session_state.messages.append({"role": "assistant", "content": response})
