# ... otros imports
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import streamlit as st
import os

# --- Módulo de Preguntas y Respuestas ---
st.header("2. Preguntas y Respuestas")

if "vector_store" not in st.session_state:
    st.warning("Por favor, procesa los documentos primero para poder hacer preguntas.")
else:
    question = st.text_input("Haz tu pregunta sobre los documentos:")
    
    if question:
        if st.button("Obtener Respuesta"):
            with st.spinner("Generando respuesta..."):
                
                # 💡 CAMBIO CLAVE: Configurar la API de Gemini
                # Asegúrate de guardar la clave en Streamlit Secrets como "GOOGLE_API_KEY"
                genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                
                # Inicializar el LLM de Gemini
                llm = ChatGoogleGenerativeAI(model="gemini-pro") # Usar gemini-pro para texto
                
                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever()
                )
                
                response = rag_chain.invoke(question)
                
                st.write("---")
                st.subheader("Respuesta:")
                st.write(response["result"])

