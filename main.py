import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import google.generativeai as genai
import random

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

def generate_questions(rag_chain, num_questions=10):
    """Genera una lista de preguntas aleatorias basadas en el contenido."""
    questions = []
    for _ in range(num_questions):
        prompt = "Genera una pregunta de una sola oración que se pueda responder directamente con el contenido de los documentos. La pregunta debe ser clara y concisa."
        try:
            question = rag_chain.invoke(prompt)
            questions.append(question["result"])
        except Exception as e:
            st.error(f"Error generando la pregunta {_ + 1}: {e}")
            return []
    return questions

def grade_answer(rag_chain, user_answer, question):
    """Evalúa la respuesta del usuario y asigna un puntaje de 0 a 5."""
    prompt = f"""
    Eres un evaluador experto. Tu tarea es calificar las respuestas de los usuarios a una pregunta basada en documentos de capacitación.

    Pregunta: "{question}"
    Respuesta del usuario: "{user_answer}"
    
    Da una calificación del 0 al 5, donde:
    5: La respuesta es excelente, completa y precisa.
    4: La respuesta es muy buena, con pequeños detalles faltantes.
    3: La respuesta es correcta, pero básica o con información incompleta.
    2: La respuesta es parcialmente correcta, pero contiene errores significativos.
    1: La respuesta es incorrecta.
    0: La respuesta no tiene relación con la pregunta.

    Además de la calificación, proporciona un feedback conciso de una oración. No incluyas la respuesta correcta en el feedback, solo guía al usuario a mejorar.
    
    Formato de la respuesta:
    Calificación: [0-5]
    Feedback: [Tu feedback conciso aquí]
    """
    try:
        evaluation = rag_chain.invoke(prompt)
        # Limpiar la respuesta para extraer la calificación
        score = 0
        feedback = "No se pudo obtener el feedback."
        lines = evaluation["result"].split('\n')
        for line in lines:
            if "Calificación:" in line:
                try:
                    score = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    score = 0
            if "Feedback:" in line:
                feedback = line.split(":", 1)[1].strip()
        return score, feedback
    except Exception as e:
        st.error(f"Error al evaluar la respuesta: {e}")
        return 0, "Error en la evaluación del sistema."

# --- Configuración y Título ---
st.set_page_config(page_title="Mentor.IA - Finanzauto", layout="wide")
st.title("Mentor.IA 🤖")

# --- Inicialización del Estado de la Sesión ---
if "temas" not in st.session_state:
    st.session_state.temas = {}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = {
        "active": False,
        "topic": "",
        "questions": [],
        "answers": [],
        "scores": [],
        "current_q_index": 0,
        "final_score": 0
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
    
    st.subheader("Reporte de Avance")
    total_temas = len(st.session_state.temas)
    temas_finalizados = sum(1 for tema in st.session_state.temas.values() if tema["evaluado"])
    porcentaje_finalizado = (temas_finalizados / total_temas) * 100 if total_temas > 0 else 0
    st.metric(label="Temas Finalizados", value=f"{temas_finalizados}/{total_temas}")
    st.progress(porcentaje_finalizado / 100, text=f"{porcentaje_finalizado:.0f}% completado")
    
    if temas_finalizados > 0:
        promedio_finalizados = sum(t["puntaje"] for t in st.session_state.temas.values() if t["evaluado"]) / temas_finalizados
        st.metric(label="Calificación Promedio", value=f"{promedio_finalizados:.1f}/5")

    st.write("---")
    st.subheader("Temas Asignados")
    for tema, data in st.session_state.temas.items():
        if not data["evaluado"]:
            st.write(f"- {tema}")
    
    st.subheader("Temas Finalizados")
    for tema, data in st.session_state.temas.items():
        if data["evaluado"]:
            st.write(f"- {tema}: **{data['puntaje']}/5**")
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

### **Módulo de Evaluación (Temas y Quizz)**
st.header("3. Escuela de Aprendizaje: Evaluación")

if st.session_state.vector_store is None:
    st.warning("Para iniciar una evaluación, por favor carga y procesa los documentos primero.")
else:
    topic_options = list(st.session_state.temas.keys())
    selected_topic = st.selectbox("Selecciona un tema para evaluar:", options=topic_options)

    if not st.session_state.current_quiz["active"]:
        if st.button("Iniciar Evaluación"):
            with st.spinner(f"Generando 10 preguntas sobre '{selected_topic}'..."):
                st.session_state.current_quiz["active"] = True
                st.session_state.current_quiz["topic"] = selected_topic
                st.session_state.current_quiz["questions"] = generate_questions(get_qa_chain(st.session_state.vector_store), num_questions=10)
                st.session_state.current_quiz["answers"] = []
                st.session_state.current_quiz["scores"] = []
                st.session_state.current_quiz["current_q_index"] = 0
                st.session_state.current_quiz["final_score"] = 0
                st.rerun()

    if st.session_state.current_quiz["active"]:
        quiz_data = st.session_state.current_quiz
        current_q_index = quiz_data["current_q_index"]

        if current_q_index < len(quiz_data["questions"]):
            st.subheader(f"Pregunta {current_q_index + 1}/10 sobre el tema: **{quiz_data['topic']}**")
            st.write(quiz_data["questions"][current_q_index])
            user_answer = st.text_area("Tu respuesta:")
            
            if st.button("Evaluar y Siguiente"):
                if not user_answer:
                    st.warning("Por favor, escribe tu respuesta antes de continuar.")
                else:
                    score, feedback = grade_answer(get_qa_chain(st.session_state.vector_store), user_answer, quiz_data["questions"][current_q_index])
                    quiz_data["answers"].append(user_answer)
                    quiz_data["scores"].append({"score": score, "feedback": feedback})
                    quiz_data["current_q_index"] += 1
                    
                    st.success(f"Calificación de la pregunta: {score}/5")
                    st.info(f"Feedback: {feedback}")
                    st.rerun()
        else:
            # Finalizar el quiz y mostrar resultados
            final_score = sum(q["score"] for q in quiz_data["scores"])
            promedio_final = final_score / len(quiz_data["scores"])

            st.subheader("🎉 ¡Evaluación Finalizada! 🎉")
            st.metric(label="Nota Final Promedio", value=f"{promedio_final:.1f}/5")
            
            if promedio_final >= 3.0:
                st.success("¡Felicidades! 🎉 Has Aprobado la evaluación.")
            else:
                st.error("Lo siento. 😔 No has aprobado la evaluación.")
                st.warning(f"Ruta de Aprendizaje Personalizada: Te recomendamos repasar el tema '{quiz_data['topic']}' y sus documentos de apoyo para mejorar tus conocimientos.")

            # Actualizar el estado del tema
            st.session_state.temas[quiz_data["topic"]]["evaluado"] = True
            st.session_state.temas[quiz_data["topic"]]["puntaje"] = promedio_final

            st.write("---")
            st.subheader("Resumen de Preguntas y Calificaciones")
            for i, q in enumerate(quiz_data["questions"]):
                st.markdown(f"**Pregunta {i+1}:** {q}")
                st.markdown(f"**Tu Respuesta:** {quiz_data['answers'][i]}")
                st.markdown(f"**Calificación:** {quiz_data['scores'][i]['score']}/5 - {quiz_data['scores'][i]['feedback']}")
                st.write("---")
            
            # Botón para salir del quiz
            if st.button("Finalizar y Volver al Menú"):
                st.session_state.current_quiz["active"] = False
                st.rerun()
