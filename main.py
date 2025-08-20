import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Título de la aplicación
st.title("Chatbot de Conocimiento RAG")

# --- Módulo de Carga de Documentos ---
st.header("1. Carga de Documentos")
st.write("Sube uno o varios archivos PDF para crear la base de conocimiento.")

uploaded_files = st.file_uploader(
    "Selecciona los archivos PDF", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    if st.button("Procesar Archivos"):
        with st.spinner("Procesando documentos..."):
            
            # Crear una carpeta temporal para los PDFs
            temp_dir = "temp_pdfs"
            os.makedirs(temp_dir, exist_ok=True)
            
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
            embeddings = OllamaEmbeddings(model="llama3")  # Asegúrate de que este modelo esté disponible en tu entorno
            
            # Creación del vector store
            vector_store = FAISS.from_documents(splits, embeddings)
            
            st.session_state.vector_store = vector_store
            st.success("Documentos procesados y base de conocimiento creada. ¡Ahora puedes hacer preguntas!")
            
            # Limpiar la carpeta temporal
            for path in file_paths:
                os.remove(path)
            os.rmdir(temp_dir)

# --- Módulo de Preguntas y Respuestas ---
st.header("2. Preguntas y Respuestas")

if "vector_store" not in st.session_state:
    st.warning("Por favor, procesa los documentos primero para poder hacer preguntas.")
else:
    question = st.text_input("Haz tu pregunta sobre los documentos:")
    
    if question:
        if st.button("Obtener Respuesta"):
            with st.spinner("Generando respuesta..."):
                
                # Inicializar el LLM de Ollama
                llm = Ollama(model="llama3")
                
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
