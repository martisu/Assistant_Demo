# RAG LLM Demo

## English
---

A demonstration of Retrieval Augmented Generation (RAG) implementation based on OpenAI GPT, designed to work with .docx and .pdf files. This application allows you to configure an AI assistant to be an expert in the domain of your provided documents.

### Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Demo Videos](#demo-videos)
- [Authors](#authors)

### Features
- Document processing for .docx and .pdf files
- Customizable AI assistant expertise
- Context-aware responses based on your documents
- User-friendly Streamlit interface
- Support for natural language queries

### Requirements
- **Python 3.8** (higher versions may not be fully tested)
- **OpenAI API Key**
- **Documents**: .docx or .pdf files containing domain knowledge

### Installation

#### 1. Clone the Repository
```bash
git clone [repository-url]
cd rag-llm-demo
```

#### 2. Create and Set Up the Virtual Environment
```bash
cd ./src/
python3 -m venv demoragenv
source demoragenv/bin/activate
pip install -r ../requirements.txt
python -m playwright install
deactivate
```

### Configuration

#### 1. API Credentials
Create a `credentials.yml` file in the `config/` directory:
```yaml
OPENAI_API_KEY: 'your_api_key'
```

#### 2. Assistant Configuration
Edit the `config/ragllm_params.yml` file to customize the assistant's role:
```yaml
assistant_role_instruction: "You are an expert in [specific domain]. Answer questions related to [specific expertise]."
```

#### 3. Add Your Documents
Place your .docx or .pdf files in the `./data/` folder. These documents will be used to provide context for the assistant's responses.

### Running the Application

```bash
# Activate the virtual environment
source src/demoragenv/bin/activate

# Launch the application
python -m streamlit run ./frontend/rag_interface.py
```

The application will open in your default web browser.

### Usage

1. **Start the application** as described above
2. **Ask questions** related to the content of your documents
3. The assistant will provide **contextually relevant answers** based on the information in your documents
4. You can **refine your queries** to get more specific information

### Project Structure

```
rag-llm-demo/
├── config/
│   ├── credentials.yml        # API credentials (create this file)
│   └── ragllm_params.yml      # Assistant configuration
├── data/                      # Place your documents here
├── frontend/
│   └── rag_interface.py       # Streamlit interface
├── src/                       # Source code
│   └── demoragenv/            # Virtual environment
├── requirements.txt           # Project dependencies
└── README.md                  # This documentation
```

### Demo Videos

- **English Demo**: [Watch on Loom](https://www.loom.com/share/6c1d2e0709c2491186cac3162b44898d?sid=9dc96895-4667-4111-be9b-d9be7fd9eeab)
- **Spanish Demo**: [Watch on Loom](https://www.loom.com/share/f4f53ece35764e1c8141267fd2ccf472?sid=71927c95-7640-4687-9207-59e55ec43f6f)

### Authors

- **Kevin Suin**: Mechanical Engineer with experience in Data Science and Machine Learning
- **Ernest Martínez**: Computer Technical Engineer specialized in SAP

### Notes

- Azure OpenAI configuration is pending implementation
- Ensure your OpenAI API key is valid and has sufficient credits
- Refer to [OpenAI Documentation](https://beta.openai.com/docs/) and [Streamlit Documentation](https://docs.streamlit.io/) for advanced configurations

*This document has been prepared by the authors to provide comprehensive guidance for implementing the RAG LLM Demo. For more information or inquiries, please contact the development team through Sopra Steria's official channels.*

---

## Español
---

Una demostración de implementación de Recuperación Aumentada de Generación (RAG) basada en OpenAI GPT, diseñada para trabajar con archivos .docx y .pdf. Esta aplicación te permite configurar un asistente de IA para que sea un experto en el dominio de tus documentos proporcionados.

### Tabla de Contenidos
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Ejecución de la Aplicación](#ejecución-de-la-aplicación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Videos de Demostración](#videos-de-demostración)
- [Autores](#autores)

### Características
- Procesamiento de documentos para archivos .docx y .pdf
- Experiencia del asistente de IA personalizable
- Respuestas contextualizadas basadas en tus documentos
- Interfaz de usuario amigable con Streamlit
- Soporte para consultas en lenguaje natural

### Requisitos
- **Python 3.8** (versiones superiores pueden no estar completamente probadas)
- **Clave de API de OpenAI**
- **Documentos**: archivos .docx o .pdf que contengan conocimiento del dominio

### Instalación

#### 1. Clonar el Repositorio
```bash
git clone [url-del-repositorio]
cd rag-llm-demo
```

#### 2. Crear y Configurar el Entorno Virtual
```bash
cd ./src/
python3 -m venv demoragenv
source demoragenv/bin/activate
pip install -r ../requirements.txt
python -m playwright install
deactivate
```

### Configuración

#### 1. Credenciales de API
Crea un archivo `credentials.yml` en el directorio `config/`:
```yaml
OPENAI_API_KEY: 'tu_clave_de_api'
```

#### 2. Configuración del Asistente
Edita el archivo `config/ragllm_params.yml` para personalizar el rol del asistente:
```yaml
assistant_role_instruction: "Eres un experto en [dominio específico]. Responde a preguntas relacionadas con [experiencia específica]."
```

#### 3. Añadir tus Documentos
Coloca tus archivos .docx o .pdf en la carpeta `./data/`. Estos documentos se utilizarán para proporcionar contexto a las respuestas del asistente.

### Ejecución de la Aplicación

```bash
# Activa el entorno virtual
source src/demoragenv/bin/activate

# Inicia la aplicación
python -m streamlit run ./frontend/rag_interface.py
```

La aplicación se abrirá en tu navegador web predeterminado.

### Uso

1. **Inicia la aplicación** como se describe arriba
2. **Haz preguntas** relacionadas con el contenido de tus documentos
3. El asistente proporcionará **respuestas contextualmente relevantes** basadas en la información de tus documentos
4. Puedes **refinar tus consultas** para obtener información más específica

### Estructura del Proyecto

```
rag-llm-demo/
├── config/
│   ├── credentials.yml        # Credenciales de API (crea este archivo)
│   └── ragllm_params.yml      # Configuración del asistente
├── data/                      # Coloca tus documentos aquí
├── frontend/
│   └── rag_interface.py       # Interfaz de Streamlit
├── src/                       # Código fuente
│   └── demoragenv/            # Entorno virtual
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Esta documentación
```

### Videos de Demostración

- **Demo en Inglés**: [Ver en Loom](https://www.loom.com/share/6c1d2e0709c2491186cac3162b44898d?sid=9dc96895-4667-4111-be9b-d9be7fd9eeab)
- **Demo en Español**: [Ver en Loom](https://www.loom.com/share/f4f53ece35764e1c8141267fd2ccf472?sid=71927c95-7640-4687-9207-59e55ec43f6f)

### Autores

- **Kevin Suin**: Ingeniero Mecánico, con experiencia en Ciencia de Datos y Machine Learning
- **Ernest Martínez**: Ingeniero Técnico Informático, especializado en SAP

### Notas

- La configuración para Azure OpenAI está pendiente de implementación
- Asegúrate de que tu clave de API de OpenAI sea válida y tenga crédito suficiente
- Consulta la [Documentación de OpenAI](https://beta.openai.com/docs/) y la [Documentación de Streamlit](https://docs.streamlit.io/) para configuraciones avanzadas

*Este documento ha sido elaborado por los autores con el fin de proporcionar una guía completa y detallada para la implementación de la demo RAG LLM. Para más información o consultas, por favor, contacte con el equipo de desarrollo a través de los canales oficiales de Sopra Steria.*
