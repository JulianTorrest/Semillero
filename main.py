import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings  # Nuevo
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama  # Mantener si tienes acceso al servidor
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
            
            temp_dir = "temp_pdfs"
            os.makedirs(temp_dir, exist_ok=True)
            
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
            
            # 💡 CAMBIO CLAVE: Usar HuggingFaceEmbeddings en lugar de OllamaEmbeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            vector_store = FAISS.from_documents(splits, embeddings)
            
            st.session_state.vector_store = vector_store
            st.success("Documentos procesados y base de conocimiento creada. ¡Ahora puedes hacer preguntas!")
            
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
                
                # ❗ AVISO: El uso de Ollama como LLM aquí aún generará un error
                # si no tienes un servidor de Ollama accesible.
                # Lo mejor es usar un servicio como OpenAI o Cohere.
                # Por ahora, mantengamos el código como está para que veas el siguiente error.
                llm = Ollama(model="llama3")
                
                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever()
                )
                
                response = rag_chain.invoke(question)
                
                st.write("---")
                st.subheader("Respuesta:")
                st.write(response["result"])

