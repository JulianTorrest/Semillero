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

def generate_questions_and_answers(vector_store, num_questions=4):
    """
    Genera preguntas y respuestas únicas y las almacena.
    Selecciona fragmentos de texto aleatorios para garantizar la unicidad.
    """
    questions = []
    answers = []
    
    all_splits = vector_store.as_retriever().get_relevant_documents("")
    if len(all_splits) < num_questions:
        st.error(f"Se necesitan al menos {num_questions} fragmentos de texto para generar el quiz, pero solo se encontraron {len(all_splits)}.")
        return [], []
    
    selected_splits = random.sample(all_splits, num_questions)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])
    
    for split in selected_splits:
        prompt_q = f"""
        Usando este fragmento de texto:
        "{split.page_content}"
        Genera una pregunta de una sola oración que se pueda responder con la información de este fragmento. La pregunta debe ser clara y concisa.
        """
        question = llm.invoke(prompt_q).content
        questions.append(question)
        
        prompt_a = f"""
        Usando este fragmento de texto:
        "{split.page_content}"
        Responde a la pregunta: "{question}"
        Tu respuesta debe ser directa, concisa y basada únicamente en el fragmento de texto proporcionado.
        """
        answer = llm.invoke(prompt_a).content
        answers.append(answer)
        
    return questions, answers

def grade_answer(rag_chain, user_answer, question, correct_answer):
    """Evalúa la respuesta del usuario y asigna un puntaje de 0 a 5."""
    prompt = f"""
    Eres un evaluador experto. Tu tarea es calificar las respuestas de los usuarios a una pregunta basada en documentos de capacitación.

    Pregunta: "{question}"
    Respuesta del usuario: "{user_answer}"
    Respuesta correcta (extraída del documento): "{correct_answer}"
    
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

def generate_learning_path_content(rag_chain, topic):
    """Genera tips y recomendaciones para un tema específico."""
    prompt = f"""
    Genera un resumen para la ruta de aprendizaje del tema '{topic}'. 
    Incluye:
    - Una sección de "Resumen de Puntos Clave" con 3-4 viñetas.
    - Una sección de "Tips y Recomendaciones" con 3 viñetas.
    - Una sección de "Siguientes Pasos" con 2 recomendaciones de mejora.
    El contenido debe ser directo, útil y basado en la información de los documentos.
    """
    try:
        content = rag_chain.invoke(prompt)
        return content["result"]
    except Exception as e:
        return f"No se pudo generar el contenido para este tema: {e}"

def grade_case_simulation(rag_chain, user_answer, scenario):
    """
    Evalúa la respuesta del usuario en una simulación de caso.
    Califica la respuesta en base a contenido, tono, empatía y profesionalismo.
    """
    prompt = f"""
    Eres un evaluador de casos de servicio al cliente. Tu tarea es calificar la respuesta del usuario a un escenario.

    Escenario: "{scenario}"
    Respuesta del usuario: "{user_answer}"

    Califica la respuesta del usuario de 0 a 5 en base a los siguientes criterios:
    1.  **Contenido y Precisión (0-5):** ¿La respuesta aborda correctamente el problema?
    2.  **Tono y Profesionalismo (0-5):** ¿El tono es apropiado y profesional?
    3.  **Empatía (0-5):** ¿La respuesta muestra comprensión y empatía hacia el cliente?

    Proporciona un feedback constructivo para cada criterio y una calificación final consolidada del 0 al 5.
    
    Formato de la respuesta:
    Calificación Final: [0-5]
    Feedback de Contenido: [Tu feedback aquí]
    Feedback de Tono: [Tu feedback aquí]
    Feedback de Empatía: [Tu feedback aquí]
    """
    try:
        evaluation = rag_chain.invoke(prompt)
        return evaluation["result"]
    except Exception as e:
        st.error(f"Error al evaluar la simulación: {e}")
        return "No se pudo obtener la evaluación."

def check_and_award_badges(username, temas, quiz_scores):
    """Verifica y otorga insignias al usuario."""
    awarded_badges = []
    
    # Insignia: Completaste un Tema
    if temas["evaluado"] and temas["puntaje"] >= 3.0:
        badge_name = f"Master en {temas['topic']}"
        if badge_name not in st.session_state.users[username]["badges"]:
            st.session_state.users[username]["badges"].append(badge_name)
            awarded_badges.append(badge_name)
            
    # Insignia: Puntaje Perfecto
    if all(s['score'] == 5 for s in quiz_scores):
        if "Genio de la Cartera 🧠" not in st.session_state.users[username]["badges"]:
            st.session_state.users[username]["badges"].append("Genio de la Cartera 🧠")
            awarded_badges.append("Genio de la Cartera 🧠")

    # Insignia: Primer Paso (al completar la primera evaluación)
    if temas["evaluado"] and len(st.session_state.users[username]["temas_completados"]) == 1:
        if "Primer Paso 👣" not in st.session_state.users[username]["badges"]:
            st.session_state.users[username]["badges"].append("Primer Paso 👣")
            awarded_badges.append("Primer Paso 👣")
            
    return awarded_badges

def load_users():
    """Simula una base de datos de usuarios con datos de ejemplo."""
    if "users" not in st.session_state:
        st.session_state.users = {
            "Julian Yamid Torres Torres": {
                "temas_completados": {
                    "Tipos de clientes y manejo": 4.5, 
                    "Negociación de pagos": 4.2
                },
                "badges": ["Master en Tipos de clientes y manejo", "Primer Paso 👣"]
            },
            "Sofia Gomez": {
                "temas_completados": {
                    "Manejo de PQRs": 4.8
                },
                "badges": ["Master en Manejo de PQRs"]
            },
            "Carlos Ramirez": {
                "temas_completados": {
                    "Procesos de cartera": 3.9
                },
                "badges": []
            }
        }
    # Asegura que los temas de los usuarios existan en st.session_state.temas
    for user, data in st.session_state.users.items():
        for tema, puntaje in data["temas_completados"].items():
            if tema not in st.session_state.temas:
                st.session_state.temas[tema] = {"evaluado": True, "puntaje": puntaje, "contenido_cargado": False}

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
        "correct_answers": [],
        "answers": [],
        "scores": [],
        "current_q_index": 0,
        "final_score": 0
    }
if "learning_paths" not in st.session_state:
    st.session_state.learning_paths = {}

load_users()

def get_topics_from_files(uploaded_files):
    """Extrae los nombres de los archivos como temas de capacitación."""
    for file in uploaded_files:
        topic_name = os.path.splitext(file.name)[0].replace("_", " ").title()
        if topic_name not in st.session_state.temas:
            st.session_state.temas[topic_name] = {"evaluado": False, "puntaje": 0, "contenido_cargado": True}
        if topic_name not in st.session_state.learning_paths:
            st.session_state.learning_paths[topic_name] = None

# --- Barra Lateral (Menú y Perfil de Usuario) ---
with st.sidebar:
    st.header("👤 Perfil de Usuario")
    st.write("---")
    current_user = "Julian Yamid Torres Torres"
    st.write(f"**Nombre:** {current_user}")
    st.subheader("Mis Insignias")
    if st.session_state.users[current_user]["badges"]:
        for badge in st.session_state.users[current_user]["badges"]:
            st.write(f"🏅 {badge}")
    else:
        st.write("Aún no tienes insignias. ¡Completa temas para ganar la primera!")
    st.write("---")

    st.header("📚 Escuela: Cartera")
    st.write("---")
    
    st.subheader("Reporte de Avance")
    temas_evaluados = [t for t in st.session_state.temas.values() if t["evaluado"]]
    total_temas = len(st.session_state.temas)
    
    if temas_evaluados:
        promedio_finalizados = sum(t["puntaje"] for t in temas_evaluados) / len(temas_evaluados)
    else:
        promedio_finalizados = 0
    st.metric(label="Calificación Promedio", value=f"{promedio_finalizados:.1f}/5")

    temas_finalizados = len(temas_evaluados)
    porcentaje_finalizado = (temas_finalizados / total_temas) * 100 if total_temas > 0 else 0
    st.metric(label="Temas Finalizados", value=f"{temas_finalizados}/{total_temas}")
    st.progress(porcentaje_finalizado / 100, text=f"{porcentaje_finalizado:.0f}% completado")

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
    st.header("🏆 Tabla de Liderazgo")
    leaderboard_data = []
    for user, data in st.session_state.users.items():
        if data["temas_completados"]:
            avg_score = sum(data["temas_completados"].values()) / len(data["temas_completados"])
            leaderboard_data.append({
                "Usuario": user,
                "Temas Completados": len(data["temas_completados"]),
                "Puntaje Promedio": f"{avg_score:.1f}"
            })
    
    leaderboard_data.sort(key=lambda x: (x["Temas Completados"], float(x["Puntaje Promedio"])), reverse=True)
    st.table(leaderboard_data)
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

# --- Módulo de Preguntas y Respuestas (Chat con Mentor.IA) ---
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

# --- Módulo de Evaluación (Temas y Quizz) ---
st.header("3. Escuela de Aprendizaje: Evaluación")
if st.session_state.vector_store is None:
    st.warning("Para iniciar una evaluación, por favor carga y procesa los documentos primero.")
else:
    topic_options = list(st.session_state.temas.keys())
    selected_topic = st.selectbox("Selecciona un tema para evaluar:", options=topic_options)

    if not st.session_state.current_quiz["active"]:
        if st.button("Iniciar Evaluación"):
            with st.spinner(f"Generando 4 preguntas sobre '{selected_topic}'..."):
                st.session_state.current_quiz["active"] = True
                st.session_state.current_quiz["topic"] = selected_topic
                questions, correct_answers = generate_questions_and_answers(st.session_state.vector_store, num_questions=4)
                
                if questions and correct_answers:
                    st.session_state.current_quiz["questions"] = questions
                    st.session_state.current_quiz["correct_answers"] = correct_answers
                    st.session_state.current_quiz["answers"] = []
                    st.session_state.current_quiz["scores"] = []
                    st.session_state.current_quiz["current_q_index"] = 0
                    st.session_state.current_quiz["final_score"] = 0
                    st.rerun()
                else:
                    st.session_state.current_quiz["active"] = False
                    st.error("No se pudieron generar las preguntas. Por favor, asegúrate de que los documentos contienen suficiente información.")

    if st.session_state.current_quiz["active"]:
        quiz_data = st.session_state.current_quiz
        current_q_index = quiz_data["current_q_index"]

        if current_q_index < len(quiz_data["questions"]):
            st.subheader(f"Pregunta {current_q_index + 1}/4 sobre el tema: **{quiz_data['topic']}**")
            st.write(quiz_data["questions"][current_q_index])
            user_answer = st.text_area("Tu respuesta:")
            
            if st.button("Evaluar y Siguiente"):
                if not user_answer:
                    st.warning("Por favor, escribe tu respuesta antes de continuar.")
                else:
                    score, feedback = grade_answer(get_qa_chain(st.session_state.vector_store), user_answer, quiz_data["questions"][current_q_index], quiz_data["correct_answers"][current_q_index])
                    quiz_data["answers"].append(user_answer)
                    quiz_data["scores"].append({"score": score, "feedback": feedback})
                    quiz_data["current_q_index"] += 1
                    
                    st.success(f"Calificación de la pregunta: {score}/5")
                    st.info(f"Feedback: {feedback}")
                    st.rerun()
        else:
            final_score = sum(q["score"] for q in quiz_data["scores"])
            promedio_final = final_score / len(quiz_data["scores"])
            st.subheader("🎉 ¡Evaluación Finalizada! 🎉")
            st.metric(label="Nota Final Promedio", value=f"{promedio_final:.1f}/5")
            
            if promedio_final >= 3.0:
                st.success("¡Felicidades! 🎉 Has Aprobado la evaluación.")
            else:
                st.error("Lo siento. 😔 No has aprobado la evaluación.")
                st.warning(f"Ruta de Aprendizaje Personalizada: Te recomendamos repasar el tema '{quiz_data['topic']}' y sus documentos de apoyo para mejorar tus conocimientos.")

            st.session_state.temas[quiz_data["topic"]]["evaluado"] = True
            st.session_state.temas[quiz_data["topic"]]["puntaje"] = promedio_final
            
            st.session_state.users[current_user]["temas_completados"][quiz_data["topic"]] = promedio_final
            
            new_badges = check_and_award_badges(current_user, {"topic": quiz_data["topic"], "evaluado": True, "puntaje": promedio_final}, quiz_data["scores"])
            if new_badges:
                st.balloons()
                st.success("¡Has ganado una nueva insignia!")
                for badge in new_badges:
                    st.write(f"**🏅 {badge}**")

            st.write("---")
            st.subheader("Resumen de Preguntas y Calificaciones")
            for i, q in enumerate(quiz_data["questions"]):
                st.markdown(f"**Pregunta {i+1}:** {q}")
                st.markdown(f"**Tu Respuesta:** {quiz_data['answers'][i]}")
                st.markdown(f"**Calificación:** {quiz_data['scores'][i]['score']}/5 - {quiz_data['scores'][i]['feedback']}")
                st.write("---")
            
            if st.button("Finalizar y Volver al Menú"):
                st.session_state.current_quiz["active"] = False
                st.rerun()

# --- Módulo de Ruta de Aprendizaje Personalizada ---
st.header("4. Ruta de Aprendizaje Personalizada")

if st.session_state.vector_store is None:
    st.warning("Para ver las rutas de aprendizaje, por favor procesa los documentos primero.")
else:
    topic_options_path = ["Selecciona un tema"] + list(st.session_state.temas.keys())
    selected_path_topic = st.selectbox("Selecciona un tema para ver su ruta de aprendizaje:", options=topic_options_path)

    if selected_path_topic != "Selecciona un tema":
        st.write("---")
        st.subheader(f"Ruta de Aprendizaje: {selected_path_topic}")
        
        if st.session_state.temas.get(selected_path_topic, {}).get("contenido_cargado") and st.session_state.vector_store is not None:
            with st.spinner("Generando contenido de la ruta de aprendizaje..."):
                rag_chain = get_qa_chain(st.session_state.vector_store)
                content = generate_learning_path_content(rag_chain, selected_path_topic)
            st.write(content)
        else:
            st.info("Este tema aún no ha sido cargado o procesado.")

# --- Módulo de Simulación de Casos ---
st.header("5. Simulación de Casos")
if st.session_state.vector_store is None:
    st.warning("Para iniciar una simulación, por favor carga y procesa los documentos primero.")
else:
    st.markdown(
    """
    <div style="background-color:#003366; padding: 20px; border-radius: 10px; color:white;">
        <p style="text-align:center; font-size:18px;">
            ¡Bienvenido a la simulación! En este espacio, pondrás a prueba tus habilidades de comunicación y resolución de problemas en escenarios reales de atención al cliente.
        </p>
        <p style="text-align:center; font-size:18px;">
            Tu tarea es describir con detalle, como si estuvieras en una situación real, cómo manejarías la situación. Mentor.IA evaluará tu respuesta no solo por la precisión de la información, sino también por tu **tono**, **empatía** y **profesionalismo**.
        </p>
        <p style="text-align:center; font-size:18px;">
            **¡Estás listo para demostrar tu potencial!**
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
    
    scenario = st.text_input("Ingresa el escenario de servicio al cliente:")
    if scenario:
        user_response = st.text_area("Describe cómo manejarías este caso:")
        if st.button("Evaluar mi Simulación"):
            if not user_response:
                st.warning("Por favor, escribe tu respuesta antes de evaluar.")
            else:
                with st.spinner("Evaluando tu respuesta..."):
                    rag_chain = get_qa_chain(st.session_state.vector_store)
                    evaluation_result = grade_case_simulation(rag_chain, user_response, scenario)
                    st.write("---")
                    st.subheader("Resultado de la Simulación")
                    st.info(evaluation_result)
