import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import google.generativeai as genai

# --- Configuración y Título ---
st.set_page_config(page_title="Mentor.IA - Finanzauto", layout="wide")
st.title("Mentor.IA 🤖")

# --- Módulo de Funciones para Temas ---
# Esta sección simula los temas y las evaluaciones.
# En una aplicación real, esta información vendría de una base de datos.
if "temas" not in st.session_state:
    st.session_state.temas = {
        "Procesos de cartera": {"evaluado": False, "puntaje": 0, "contenido_cargado": False},
        "Tipos de clientes y manejo": {"evaluado": True, "puntaje": 95, "contenido_cargado": True},
        "Manejo de PQRs": {"evaluado": False, "puntaje": 0, "contenido_cargado": False},
        "Negociación de pagos": {"evaluado": True, "puntaje": 88, "contenido_cargado": True}
    }

def get_topics_from_files(uploaded_files):
    """Extrae los nombres de los archivos como temas de capacitación."""
    temas_nuevos = {}
    for file in uploaded_files:
        topic_name = os.path.splitext(file.name)[0].replace("_", " ").title()
        temas_nuevos[topic_name] = {"evaluado": False, "puntaje": 0, "contenido_cargado": True}
    return temas_nuevos

# --- Barra Lateral (Menú y Perfil de Usuario) ---
with st.sidebar:
    st.header("👤 Perfil de Usuario")
    st.write("---")
    st.write("**Nombre:** Julian Yamid Torres Torres")
    st.write("---")

    st.header("📚 Escuela: Cartera")
    st.write("---")
    st.subheader("Temas Asignados")
    for tema, data in st.session_state.temas.items():
        if not data["evaluado"]:
            st.write(f"- {tema}")
    
    st.subheader("Temas Finalizados")
    for tema, data in st.session_state.temas.items():
        if data["evaluado"]:
            st.write(f"- {tema}: **{data['puntaje']}/100**")
    st.write("---")

# --- Sección de Bienvenida y Propósito ---
st.markdown(
    """
    <div style="background-color:#003366; padding: 20px; border-radius: 10px; color:white;">
        <h2 style="color:white; text-align:center;">
            En Finanzauto creemos en el poder del aprendizaje continuo.
        </h2>
        <p style="text-align:center; font-size:18px;">
            Por eso, creamos <b>Mentor.IA</b>, una plataforma innovadora que combina teoría, práctica e inteligencia artificial para potenciar tu desarrollo profesional. Tu Escuela de Aprendizaje será: <b>“Escuela DataPro”</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# --- Módulo de Carga de Documentos ---
st.header("1. Carga de Documentos")
st.write("Sube uno o varios archivos PDF para crear la base de conocimiento.")

uploaded_files = st.file_uploader(
    "Selecciona los archivos PDF", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    if st.button("Procesar Archivos"):
        with st.spinner("Procesando documentos..."):
            try:
                # Actualizar temas con los archivos subidos
                nuevos_temas = get_topics_from_files(uploaded_files)
                st.session_state.temas.update(nuevos_temas)

                # Crear una carpeta temporal para los PDFs
                temp_dir = "temp_pdfs"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                docs = []
                for path in file_paths:
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                
                vector_store = FAISS.from_documents(splits, embeddings)
                
                st.session_state.vector_store = vector_store
                st.success("✅ Documentos procesados y base de conocimiento creada. ¡Ahora puedes hacer preguntas!")
                
            except Exception as e:
                st.error(f"❌ Ocurrió un error durante el procesamiento: {e}")
            finally:
                if os.path.exists(temp_dir):
                    for path in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, path))
                    os.rmdir(temp_dir)

# --- Módulo de Preguntas y Respuestas (Chat con Mentor.IA) ---
st.header("2. Preguntas y Respuestas")

if "vector_store" not in st.session_state:
    st.warning("Por favor, procesa los documentos primero para poder hacer preguntas.")
else:
    question = st.text_input("Haz tu pregunta sobre los documentos:")
    
    if question:
        if st.button("Obtener Respuesta"):
            with st.spinner("Mentor.IA está generando la respuesta..."):
                try:
                    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                    
                    # Usar el modelo 'gemini-2.0-flash'
                    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash") 
                    
                    rag_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever()
                    )
                    
                    response = rag_chain.invoke(question)
                    
                    st.write("---")
                    st.subheader("Respuesta de Mentor.IA:")
                    st.write(response["result"])
                    
                except Exception as e:
                    st.error(f"❌ Ocurrió un error al obtener la respuesta: {e}")
