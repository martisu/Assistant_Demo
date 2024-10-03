![Sopra Steria Logo](https://www.soprasteria.com/ResourcePackages/Bootstrap4/assets/dist/logos/logo-soprasteria.svg)

# RAG LLM Demo

Demo para la implementación de Retrieval Augmented Generation (RAG) basada en OpenAI GPT, diseñada para la utilización de archivos .docx y .pdf. Configura el tipo de asistente para que sea un experto en el dominio de datos proporcionados.

## Instalación, Configuración y Ejecución de la Aplicación

### Requisitos
- **Python 3.8** (versiones superiores pueden no estar completamente probadas).
- **Clave de API de OpenAI** (`OPENAI_API_KEY`). 
  - **Nota:** Configuración para Azure OpenAI pendiente.
- **Documentos:** Añade tus archivos .docx o .pdf en la carpeta `./data/`.

### Creación del Entorno Virtual

1. Navega al directorio `src/`:
    ```bash
    cd ./src/
    ```
2. Crea un entorno virtual:
    ```bash
    python3 -m venv demoragenv
    ```
3. Activa el entorno virtual:
    ```bash
    source demoragenv/bin/activate
    ```
4. Instala las dependencias necesarias:
    ```bash
    pip install -r ../requirements.txt
    ```
5. Desactiva el entorno virtual después de la instalación:
    ```bash
    deactivate
    ```

### Configuración

1. **Configura las Credenciales:**
   - Crea un archivo `credentials.yml` en el directorio `config/` con el siguiente formato:
     ```yaml
     OPENAI_API_KEY: 'tu_clave_de_api'
     ```
2. **Configura el Rol del Asistente:**
   - Edita el archivo `config/ragllm_params.yml` para definir los parámetros del asistente. Personaliza el campo `assistant_role_instruction` para definir el conocimiento especializado del asistente. Por ejemplo:
     ```yaml
     assistant_role_instruction: "Eres un experto en la planificacion de obras de reparacion y mantenimiento en casa. Responde a preguntas relacionadas con reparacion y mantenimiento de casas."
     ```

### Gestión de Datos

- Añade tus archivos .docx o .pdf en la carpeta `./data/`. Estos documentos serán utilizados para contextualizar y personalizar las respuestas del asistente.

### Ejecución de la Demo

1. Activa el entorno virtual:
    ```bash
    source src/demoragenv/bin/activate
    ```
2. Inicia la aplicación con Streamlit:
    ```bash
    python -m streamlit run ./frontend/rag_interface.py
    ```

Esto abrirá la interfaz de usuario en tu navegador predeterminado. Podrás interactuar con el asistente, que utilizará los documentos cargados en la carpeta `./data/` para responder a tus preguntas de forma precisa y contextualizada.

## Notas Adicionales
- Asegúrate de que la clave de API de OpenAI es válida y tiene suficiente crédito para ejecutar la demo.
- La configuración de conexión con Azure OpenAI está pendiente de implementación. Por favor, consulta la documentación de OpenAI para detalles adicionales.
- Revisa la documentación oficial de [OpenAI](https://beta.openai.com/docs/) y [Streamlit](https://docs.streamlit.io/) para configuraciones avanzadas y opciones de personalización.

**Autores:**
- Javier San Juan, Data Science en Sopra Steria. Especialista en Inteligencia Artificial y Machine Learning.
- Marlon Cárdenas, Data Science en Sopra Steria. Especialista en Inteligencia Artificial y Machine Learning.

<sub>**Nota:** Este documento ha sido elaborado por los autores con el fin de proporcionar una guía completa y detallada para la implementación de la demo RAG LLM. Para más información o consultas, por favor, contacte con el equipo de desarrollo a través de los canales oficiales de Sopra Steria.</sub>


