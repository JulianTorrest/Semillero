import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import google.generativeai as genai

# Título de la aplicación
st.title("🤖 Chatbot de Conocimiento RAG con Gemini")

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
                # Crear una carpeta temporal para los PDFs
                temp_dir = "temp_pdfs"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                # Guardar archivos subidos en la carpeta temporal
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                # Cargar documentos
                docs = []
                for path in file_paths:
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                
                # División de documentos
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # Generación de embeddings
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                
                # Creación del vector store
                vector_store = FAISS.from_documents(splits, embeddings)
                
                st.session_state.vector_store = vector_store
                st.success("✅ Documentos procesados y base de conocimiento creada. ¡Ahora puedes hacer preguntas!")
                
            except Exception as e:
                st.error(f"❌ Ocurrió un error durante el procesamiento: {e}")
            finally:
                # Limpiar la carpeta temporal
                if os.path.exists(temp_dir):
                    for path in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, path))
                    os.rmdir(temp_dir)

# ---
### Módulo de Preguntas y Respuestas

st.header("2. Preguntas y Respuestas")

if "vector_store" not in st.session_state:
    st.warning("Por favor, procesa los documentos primero para poder hacer preguntas.")
else:
    question = st.text_input("Haz tu pregunta sobre los documentos:")
    
    if question:
        if st.button("Obtener Respuesta"):
            with st.spinner("Generando respuesta..."):
                try:
                    # Configurar la API de Gemini
                    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                    
                    # Inicializar el LLM de Gemini
                    llm = ChatGoogleGenerativeAI(model="gemini-pro")
                    
                    # Crear la cadena RAG
                    rag_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever()
                    )
                    
                    # Obtener la respuesta
                    response = rag_chain.invoke(question)
                    
                    # Mostrar la respuesta
                    st.write("---")
                    st.subheader("Respuesta:")
                    st.write(response["result"])
                    
                except Exception as e:
                    st.error(f"❌ Ocurrió un error al obtener la respuesta: {e}")
