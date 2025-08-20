import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import google.generativeai as genai

# --- Funciones Auxiliares ---
def get_qa_chain(vector_store, model_name="gemini-2.0-flash"):
    """Crea y retorna la cadena RAG para preguntas y respuestas."""
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    llm = ChatGoogleGenerativeAI(model=model_name)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return rag_chain

def generate_question(rag_chain):
    """Genera una pregunta aleatoria basada en el contenido del vector store."""
    prompt = "Genera una pregunta de una sola oración que se pueda responder directamente con el contenido de los documentos. La pregunta debe ser clara y concisa."
    question = rag_chain.invoke(prompt)
    return question["result"]

def grade_answer(rag_chain, user_answer, correct_answer):
    """Evalúa la respuesta del usuario usando el LLM y da un puntaje."""
    prompt = f"""
    Evalúa la siguiente respuesta del usuario. Compara su respuesta con la respuesta correcta basada en los documentos.
    Respuesta del usuario: "{user_answer}"
    Respuesta correcta (basada en el documento): "{correct_answer}"

    Asigna una calificación del 0 al 100 y proporciona un feedback conciso de una oración.
    Formato de la respuesta:
    Calificación: [0-100]
    Feedback: [Tu feedback aquí]
    """
    evaluation = rag_chain.invoke(prompt)
    return evaluation["result"]

# --- Configuración y Título ---
st.set_page_config(page_title="Mentor.IA - Finanzauto", layout="wide")
st.title("Mentor.IA 🤖")

# --- Módulo de Funciones para Temas y Estado ---
if "temas" not in st.session_state:
    st.session_state.temas = {}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "active": False,
        "question": "",
        "correct_answer": "",
        "topic": ""
    }

def get_topics_from_files(uploaded_files):
    """Extrae los nombres de los archivos como temas de capacitación."""
    for file in uploaded_files:
        topic_name = os.path.splitext(file.name)[0].replace("_", " ").title()
        if topic_name not in st.session_state.temas:
            st.session_state.temas[topic_name] = {"evaluado": False, "puntaje": 0, "contenido_cargado": True}

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
                get_topics_from_files(uploaded_files)
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

### **Módulo de Preguntas y Respuestas (Chat con Mentor.IA)**
st.header("2. Preguntas y Respuestas")

if st.session_state.vector_store is None:
    st.warning("Por favor, procesa los documentos primero para poder usar Mentor.IA.")
else:
    question = st.text_input("Haz tu pregunta a Mentor.IA:")
    if question:
        if st.button("Obtener Respuesta"):
            with st.spinner("Mentor.IA está generando la respuesta..."):
                try:
                    rag_chain = get_qa_chain(st.session_state.vector_store)
                    response = rag_chain.invoke(question)
                    st.write("---")
                    st.subheader("Respuesta de Mentor.IA:")
                    st.write(response["result"])
                except Exception as e:
                    st.error(f"❌ Ocurrió un error al obtener la respuesta: {e}")

---

### **Módulo de Evaluación (Temas y Quizz)**
st.header("3. Escuela de Aprendizaje: Evaluación")

if st.session_state.vector_store is None:
    st.warning("Para iniciar una evaluación, por favor carga y procesa los documentos primero.")
else:
    topic_options = list(st.session_state.temas.keys())
    selected_topic = st.selectbox("Selecciona un tema para evaluar:", options=topic_options)

    if st.button("Iniciar Evaluación"):
        with st.spinner(f"Generando pregunta sobre '{selected_topic}'..."):
            try:
                # Generar una pregunta y la respuesta correcta
                rag_chain = get_qa_chain(st.session_state.vector_store)
                new_question = generate_question(rag_chain)
                
                # Para obtener la respuesta correcta, hacemos una segunda llamada
                correct_answer_chain = RetrievalQA.from_chain_type(
                    llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever()
                )
                correct_answer = correct_answer_chain.invoke(new_question)
                
                # Actualizar el estado del quiz
                st.session_state.quiz_state["active"] = True
                st.session_state.quiz_state["question"] = new_question
                st.session_state.quiz_state["correct_answer"] = correct_answer["result"]
                st.session_state.quiz_state["topic"] = selected_topic
                st.rerun() 
            except Exception as e:
                st.error(f"❌ Ocurrió un error al generar la pregunta: {e}")

    # Mostrar la pregunta y el campo de respuesta si el quiz está activo
    if st.session_state.quiz_state["active"]:
        st.subheader(f"Pregunta sobre el tema: **{st.session_state.quiz_state['topic']}**")
        st.write(st.session_state.quiz_state["question"])
        user_answer = st.text_area("Tu respuesta:")
        
        if st.button("Evaluar Respuesta"):
            if not user_answer:
                st.warning("Por favor, escribe tu respuesta antes de evaluar.")
            else:
                with st.spinner("Evaluando tu respuesta..."):
                    try:
                        rag_chain = get_qa_chain(st.session_state.vector_store)
                        evaluation = grade_answer(rag_chain, user_answer, st.session_state.quiz_state["correct_answer"])
                        st.subheader("Resultado de la Evaluación")
                        st.write(evaluation)
                        
                        # Simular el guardado del puntaje para el tema
                        if "Calificación:" in evaluation:
                            score_str = evaluation.split("Calificación: ")[1].split("\n")[0].strip()
                            try:
                                score = int(score_str)
                                st.session_state.temas[st.session_state.quiz_state["topic"]]["evaluado"] = True
                                st.session_state.temas[st.session_state.quiz_state["topic"]]["puntaje"] = score
                            except ValueError:
                                st.warning("No se pudo extraer la calificación. Por favor, revisa el formato de la respuesta.")
                        
                        st.session_state.quiz_state["active"] = False
                        st.session_state.quiz_state["question"] = ""
                        st.session_state.quiz_state["correct_answer"] = ""
                        st.session_state.quiz_state["topic"] = ""
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Ocurrió un error al evaluar la respuesta: {e}")
