import streamlit as st
import os
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import google.generativeai as genai
import random
import pandas as pd
from PIL import Image
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_to_json

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

    Proporciona un feedback constructivo para cada criterio y una calificación final consolidada del 0 a 5.
    
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
    if temas["evaluado"] and sum(len(escuelas) for escuelas in st.session_state.users[username]["temas_completados"].values()) == 1:
        if "Primer Paso 👣" not in st.session_state.users[username]["badges"]:
            st.session_state.users[username]["badges"].append("Primer Paso 👣")
            awarded_badges.append("Primer Paso 👣")
            
    # Insignia: Leyenda (alcanzar un nivel alto)
    if st.session_state.users[username]["nivel"] >= 3:
        if "Leyenda de Mentor.IA ✨" not in st.session_state.users[username]["badges"]:
            st.session_state.users[username]["badges"].append("Leyenda de Mentor.IA ✨")
            awarded_badges.append("Leyenda de Mentor.IA ✨")

    return awarded_badges

def update_user_points(username, points_to_add):
    """Actualiza los puntos y el nivel del usuario."""
    st.session_state.users[username]["puntos"] += points_to_add
    
    # Lógica de niveles (ejemplo simple)
    if st.session_state.users[username]["puntos"] >= 300 and st.session_state.users[username]["nivel"] < 3:
        st.session_state.users[username]["nivel"] = 3
        st.success(f"¡Felicidades, {username}! Has alcanzado el nivel 3. Revisa tus insignias.")
    elif st.session_state.users[username]["puntos"] >= 100 and st.session_state.users[username]["nivel"] < 2:
        st.session_state.users[username]["nivel"] = 2
        st.success(f"¡Felicidades, {username}! Has alcanzado el nivel 2.")
    elif st.session_state.users[username]["puntos"] >= 20 and st.session_state.users[username]["nivel"] < 1:
        st.session_state.users[username]["nivel"] = 1
        st.success(f"¡Felicidades, {username}! Has alcanzado el nivel 1.")


def load_users():
    """Simula una base de datos de usuarios con datos de ejemplo."""
    if "users" not in st.session_state:
        st.session_state.users = {
            "Julian Yamid Torres Torres": {
                "puntos": 150,
                "nivel": 2,
                "temas_completados": {
                    "Escuela DataPro": {
                        "Tipos de clientes y manejo": 4.5,
                        "Negociación de pagos": 4.2
                    }
                },
                "badges": ["Master en Tipos de clientes y manejo", "Primer Paso 👣"]
            },
            "Sofia Gomez": {
                "puntos": 80,
                "nivel": 1,
                "temas_completados": {
                    "Escuela de Cobradores": {
                        "Manejo de PQRs": 4.8
                    }
                },
                "badges": ["Master en Manejo de PQRs"]
            },
            "Carlos Ramirez": {
                "puntos": 25,
                "nivel": 1,
                "temas_completados": {
                    "Escuela de Verificadores": {
                        "Procesos de cartera": 3.9
                    }
                },
                "badges": []
            }
        }
    
    # Sincronizar temas completados con la nueva estructura
    for user, data in st.session_state.users.items():
        if isinstance(data.get("temas_completados"), dict):
            for escuela, temas in data["temas_completados"].items():
                if isinstance(temas, dict):
                    for tema, puntaje in temas.items():
                        if escuela in st.session_state.escuelas and tema in st.session_state.escuelas[escuela]:
                            st.session_state.escuelas[escuela][tema]["evaluado"] = True
                            st.session_state.escuelas[escuela][tema]["puntaje"] = puntaje

def get_topics_from_files(uploaded_files, school_name="Escuela DataPro"):
    """Extrae los nombres de los archivos como temas de capacitación y los asigna a una escuela."""
    for file in uploaded_files:
        topic_name = os.path.splitext(file.name)[0].replace("_", " ").title()
        if school_name not in st.session_state.escuelas:
            st.session_state.escuelas[school_name] = {}
        if topic_name not in st.session_state.escuelas[school_name]:
            st.session_state.escuelas[school_name][topic_name] = {"evaluado": False, "puntaje": 0, "contenido_cargado": True}
        else:
            st.session_state.escuelas[school_name][topic_name]["contenido_cargado"] = True

# --- Funciones de Notificaciones ---
def send_completion_notification(username, topic, score):
    """Simula el envío de una notificación por correo electrónico."""
    st.success(f"📧 ¡Notificación enviada! Se ha notificado a {username} sobre la finalización de la evaluación en '{topic}' con una nota de {score:.1f}/5.")

def generate_interactive_quiz():
    """Genera y maneja un quiz interactivo de arrastrar y soltar."""
    st.subheader("🛠️ Quiz Interactivo: Organiza el Proceso")
    st.write("Arrastra y suelta los pasos en el orden correcto para simular un proceso.")
    
    # Definir el proceso y el orden correcto
    proceso_correcto = [
        "1. Identificar el cliente",
        "2. Evaluar la situación financiera",
        "3. Proponer un acuerdo de pago",
        "4. Dar seguimiento al acuerdo",
        "5. Cerrar el caso"
    ]
    
    # Mezclar los pasos para que el usuario los ordene
    proceso_mezclado = random.sample(proceso_correcto, len(proceso_correcto))
    
    st.session_state["interactive_quiz_order"] = proceso_mezclado
    st.session_state["correct_order"] = proceso_correcto

    # Mostrar los elementos para arrastrar y soltar
    for i, item in enumerate(st.session_state["interactive_quiz_order"]):
        st.markdown(f"**{i+1}.** {item}")

    st.write("---")
    st.markdown("Ahora, arrastra y suelta los elementos en el orden correcto. (En una implementación real, esto se haría con una biblioteca de arrastrar y soltar como `st_on_hover_tabs` o similar)")

    # Simulación de la respuesta del usuario con un formulario
    with st.form("interactive_quiz_form"):
        st.write("Escribe el número del paso y el texto del paso para simular tu ordenación.")
        user_order = st.text_area("Ordena los pasos (uno por línea):", value="\n".join(proceso_mezclado))
        submitted = st.form_submit_button("Verificar Orden")
    
    if submitted:
        user_list = [line.strip() for line in user_order.split('\n') if line.strip()]
        
        if user_list == proceso_correcto:
            st.success("¡Felicidades! 🎉 Has ordenado los pasos correctamente.")
            update_user_points(st.session_state.current_user, 25)  # Sumar puntos por completar el quiz
            check_and_award_badges(st.session_state.current_user, {"topic": "Quiz Interactivo", "evaluado": True, "puntaje": 5.0}, [{"score": 5}])
        else:
            st.error("Orden incorrecto. Por favor, intenta de nuevo.")

# --- Configuración y Título ---
st.set_page_config(page_title="Mentor.IA - Finanzauto", layout="wide")
st.title("Mentor.IA 🤖")

# --- Inicialización del Estado de la Sesión ---
if "escuelas" not in st.session_state:
    st.session_state.escuelas = {
        "Escuela DataPro": {
            "Tipos de clientes y manejo": {"evaluado": False, "puntaje": 0},
            "Negociación de pagos": {"evaluado": False, "puntaje": 0},
            "Recuperación de cartera": {"evaluado": False, "puntaje": 0},
        },
        "Escuela de Cobradores": {
            "Técnicas de persuasión": {"evaluado": False, "puntaje": 0},
            "Manejo de objeciones": {"evaluado": False, "puntaje": 0},
            "Cierre de acuerdos de pago": {"evaluado": False, "puntaje": 0},
        },
        "Escuela de Verificadores": {
            "Validación de datos": {"evaluado": False, "puntaje": 0},
            "Normatividad vigente": {"evaluado": False, "puntaje": 0},
        },
        "Escuela de Atención al Cliente": {
            "Protocolo de llamadas": {"evaluado": False, "puntaje": 0},
            "Solución de conflictos": {"evaluado": False, "puntaje": 0},
        }
    }
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = {
        "active": False,
        "school": "",
        "topic": "",
        "questions": [],
        "correct_answers": [],
        "answers": [],
        "scores": [],
        "current_q_index": 0,
        "final_score": 0
    }
if "current_user" not in st.session_state:
    st.session_state.current_user = "Julian Yamid Torres Torres"


load_users()

# --- Contenido Principal con Pestañas ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["👤 Perfil", "🎓 Escuelas", "📚 Evaluación", "👨‍💼 Gestionar Usuario", "🛠️ Modificar Escuela", "📊 Analíticas"])

with tab1:
    # --- Perfil de Usuario ---
    st.header("👤 Perfil de Usuario")
    st.write("---")
    current_user = st.session_state.current_user
    st.write(f"**Nombre:** {current_user}")

    st.subheader("Estado de Nivel")
    total_temas_completados = sum(len(temas) for temas in st.session_state.users[current_user]["temas_completados"].values())
    st.metric(label="Módulos Completados", value=f"{total_temas_completados}")

    total_puntaje = 0
    total_temas = 0
    for escuela, temas in st.session_state.users[current_user]["temas_completados"].items():
        total_puntaje += sum(temas.values())
        total_temas += len(temas)
    
    promedio_general = total_puntaje / total_temas if total_temas > 0 else 0
    st.metric(label="Puntaje Promedio General", value=f"{promedio_general:.1f}/5")

    st.write("---")
    st.subheader("Gamificación")
    st.metric(label="Puntos Totales", value=st.session_state.users[current_user]["puntos"])
    st.metric(label="Nivel Actual", value=st.session_state.users[current_user]["nivel"])

    st.write("---")
    st.subheader("Mis Insignias")
    if st.session_state.users[current_user]["badges"]:
        for badge in st.session_state.users[current_user]["badges"]:
            st.write(f"🏅 {badge}")
    else:
        st.write("Aún no tienes insignias. ¡Completa módulos para ganar la primera!")
    
    st.write("---")
    st.header("🏆 Tabla de Liderazgo")
    leaderboard_data = []
    for user, data in st.session_state.users.items():
        total_temas_completados = sum(len(escuelas) for escuelas in data["temas_completados"].values())
        total_puntaje = sum(sum(temas.values()) for temas in data["temas_completados"].values())
        promedio_final = total_puntaje / total_temas_completados if total_temas_completados > 0 else 0
        leaderboard_data.append({
            "Usuario": user,
            "Puntos": data.get("puntos", 0),
            "Nivel": data.get("nivel", 0),
            "Módulos Completados": total_temas_completados,
            "Puntaje Promedio": f"{promedio_final:.1f}"
        })
    
    leaderboard_data.sort(key=lambda x: (x["Puntos"], x["Nivel"], float(x["Puntaje Promedio"])), reverse=True)
    st.table(leaderboard_data)

with tab2:
    # --- Módulo de Escuelas ---
    st.header("🎓 Escuelas de Aprendizaje")
    st.write("---")

    for escuela_nombre, temas in st.session_state.escuelas.items():
        st.subheader(f"✅ {escuela_nombre}")
        total_temas_escuela = len(temas)
        temas_completados_escuela = [t for t in temas.values() if t["evaluado"]]
        num_temas_completados = len(temas_completados_escuela)
        
        st.progress(num_temas_completados / total_temas_escuela, text=f"{num_temas_completados}/{total_temas_escuela} Módulos completados")
        
        for tema, data in temas.items():
            if data["evaluado"]:
                st.write(f"- **{tema}**: **{data['puntaje']:.1f}/5**")
            else:
                st.write(f"- **{tema}**: Pendiente")
        st.write("---")

with tab3:
    # --- Contenido de la pestaña de Evaluación ---
    st.header("1. Carga de Documentos")
    st.write("Sube archivos en formato **PDF, DOCX, XLSX, PPTX o imágenes (.png, .jpg)** para crear la base de conocimiento.")
    
    uploaded_files = st.file_uploader(
        "Selecciona los archivos", 
        type=["pdf", "docx", "pptx", "xlsx", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Procesar Archivos"):
            with st.spinner("Procesando documentos..."):
                try:
                    all_text = ""
                    for uploaded_file in uploaded_files:
                        file_type = uploaded_file.type
                        
                        if file_type == "application/pdf":
                            # Procesar PDF
                            temp_dir = "temp_files"
                            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            loader = PyPDFLoader(file_path)
                            docs = loader.load()
                            for doc in docs:
                                all_text += doc.page_content
                            os.remove(file_path)
                            st.success(f"✅ Documento PDF '{uploaded_file.name}' procesado.")

                        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.openxmlformats-officedocument.presentationml.presentation"]:
                            # Procesar DOCX y PPTX
                            elements = partition(file=uploaded_file)
                            text_content = "\n\n".join([str(e) for e in elements])
                            all_text += text_content
                            st.success(f"✅ Documento de texto '{uploaded_file.name}' procesado.")

                        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                            # Procesar XLSX
                            df = pd.read_excel(uploaded_file, sheet_name=None)
                            excel_text = ""
                            for sheet_name, sheet_df in df.items():
                                excel_text += f"\n--- Datos de la Hoja: {sheet_name} ---\n"
                                excel_text += sheet_df.to_string(index=False)
                                excel_text += "\n"
                            all_text += excel_text
                            st.success(f"✅ Documento de Excel '{uploaded_file.name}' procesado.")
                        
                        elif file_type in ["image/png", "image/jpeg"]:
                            # Procesar Imágenes (usando la capacidad de visión de Gemini)
                            st.info(f"Procesando imagen: {uploaded_file.name}")
                            img_bytes = uploaded_file.read()
                            img_model = genai.GenerativeModel('gemini-1.5-pro-latest')
                            
                            # Cargar imagen y convertirla a formato compatible
                            img = Image.open(io.BytesIO(img_bytes))
                            
                            prompt_img = """
                            Describe en detalle el contenido de esta imagen para que pueda ser utilizado como texto en una base de conocimiento. Incluye todo el texto visible, tablas, gráficos y cualquier otra información relevante.
                            """
                            response = img_model.generate_content([prompt_img, img])
                            all_text += response.text
                            st.success(f"✅ Imagen '{uploaded_file.name}' procesada.")

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_text(all_text)
                    
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    embeddings = HuggingFaceEmbeddings(model_name=model_name)
                    
                    vector_store = FAISS.from_texts(splits, embeddings)
                    st.session_state.vector_store = vector_store
                    st.success("✅ Base de conocimiento actualizada. ¡Ahora puedes hacer preguntas!")
                    
                except Exception as e:
                    st.error(f"❌ Ocurrió un error durante el procesamiento: {e}")
                finally:
                    # Limpiar el directorio temporal
                    pass

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

    st.header("3. Escuela de Aprendizaje: Evaluación")
    if st.session_state.vector_store is None:
        st.warning("Para iniciar una evaluación, por favor carga y procesa los documentos primero.")
    else:
        school_options = list(st.session_state.escuelas.keys())
        selected_school = st.selectbox("Selecciona una escuela para evaluar:", options=school_options)

        topic_options = list(st.session_state.escuelas[selected_school].keys())
        selected_topic = st.selectbox("Selecciona un tema para evaluar:", options=topic_options)

        if not st.session_state.current_quiz["active"]:
            if st.button("Iniciar Evaluación"):
                with st.spinner(f"Generando 4 preguntas sobre '{selected_topic}'..."):
                    st.session_state.current_quiz["active"] = True
                    st.session_state.current_quiz["school"] = selected_school
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
                    update_user_points(st.session_state.current_user, 20) # Añadir puntos por aprobar
                    send_completion_notification(st.session_state.current_user, quiz_data["topic"], promedio_final)
                else:
                    st.error("Lo siento. 😔 No has aprobado la evaluación.")
                    st.warning(f"Ruta de Aprendizaje Personalizada: Te recomendamos repasar el tema '{quiz_data['topic']}' y sus documentos de apoyo para mejorar tus conocimientos.")
                    update_user_points(st.session_state.current_user, 5) # Puntos de participación
                    
                st.session_state.escuelas[quiz_data["school"]][quiz_data["topic"]]["evaluado"] = True
                st.session_state.escuelas[quiz_data["school"]][quiz_data["topic"]]["puntaje"] = promedio_final
                
                if quiz_data["school"] not in st.session_state.users[st.session_state.current_user]["temas_completados"]:
                    st.session_state.users[st.session_state.current_user]["temas_completados"][quiz_data["school"]] = {}
                st.session_state.users[st.session_state.current_user]["temas_completados"][quiz_data["school"]][quiz_data["topic"]] = promedio_final
                
                new_badges = check_and_award_badges(st.session_state.current_user, {"topic": quiz_data["topic"], "evaluado": True, "puntaje": promedio_final}, quiz_data["scores"])
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

    st.header("4. Ruta de Aprendizaje Personalizada")

    if st.session_state.vector_store is None:
        st.warning("Para ver las rutas de aprendizaje, por favor procesa los documentos primero.")
    else:
        school_options_path = ["Selecciona una escuela"] + list(st.session_state.escuelas.keys())
        selected_school_path = st.selectbox("Selecciona una escuela para la ruta:", options=school_options_path)

        if selected_school_path != "Selecciona una escuela":
            topic_options_path = ["Selecciona un tema"] + list(st.session_state.escuelas[selected_school_path].keys())
            selected_path_topic = st.selectbox("Selecciona un tema para ver su ruta de aprendizaje:", options=topic_options_path)
            
            if selected_path_topic != "Selecciona un tema":
                st.write("---")
                st.subheader(f"Ruta de Aprendizaje: {selected_path_topic}")
                
                with st.spinner("Generando contenido de la ruta de aprendizaje..."):
                    rag_chain = get_qa_chain(st.session_state.vector_store)
                    content = generate_learning_path_content(rag_chain, selected_path_topic)
                st.write(content)

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
                        update_user_points(st.session_state.current_user, 10) # Puntos por simulación

    # Sección de Módulo Interactivo
    st.header("6. Módulo Interactivo")
    generate_interactive_quiz()

with tab4:
    st.header("👨‍💼 Gestionar Usuario")
    st.write("Completa el siguiente formulario para agregar un nuevo usuario.")
    
    with st.form("add_user_form"):
        st.subheader("Datos del Nuevo Usuario")
        nombre_usuario = st.text_input("Nombre Completo:")
        cedula_usuario = st.text_input("Cédula:")
        correo_usuario = st.text_input("Correo:")
        empresa_usuario = st.text_input("Empresa:")
        area_usuario = st.text_input("Área:")
        direccion_usuario = st.text_input("Dirección:")
        rol_usuario = st.text_input("Rol:")
        
        submitted = st.form_submit_button("Agregar Usuario")
        
        if submitted:
            if nombre_usuario and cedula_usuario and correo_usuario:
                new_user_data = {
                    "empresa": empresa_usuario,
                    "cedula": cedula_usuario,
                    "nombre": nombre_usuario,
                    "correo": correo_usuario,
                    "direccion": direccion_usuario,
                    "area": area_usuario,
                    "rol": rol_usuario,
                    "puntos": 0,
                    "nivel": 0,
                    "temas_completados": {},
                    "badges": []
                }
                
                st.session_state.users[nombre_usuario] = new_user_data
                st.success(f"✅ Usuario '{nombre_usuario}' agregado exitosamente.")
                st.balloons()
            else:
                st.error("Por favor, completa los campos obligatorios: Nombre, Cédula y Correo.")

with tab5:
    st.header("🛠️ Modificar Escuela")
    st.write("Selecciona una escuela para gestionar sus módulos.")

    school_to_modify = st.selectbox("Seleccionar Escuela:", list(st.session_state.escuelas.keys()))

    if school_to_modify:
        st.subheader(f"Módulos de la Escuela: {school_to_modify}")

        # Formulario para agregar un nuevo módulo
        with st.form("add_module_form", clear_on_submit=True):
            st.markdown("#### **Agregar Nuevo Módulo**")
            new_module_name = st.text_input("Nombre del Módulo:")
            new_module_duration = st.text_input("Duración (ej. 2 horas):")
            
            if st.form_submit_button("➕ Agregar Módulo"):
                if new_module_name:
                    if new_module_name not in st.session_state.escuelas[school_to_modify]:
                        st.session_state.escuelas[school_to_modify][new_module_name] = {
                            "evaluado": False, 
                            "puntaje": 0, 
                            "duracion": new_module_duration
                        }
                        st.success(f"Módulo '{new_module_name}' agregado a la escuela '{school_to_modify}'.")
                    else:
                        st.warning(f"El módulo '{new_module_name}' ya existe en esta escuela.")
                    st.rerun()
                else:
                    st.error("Por favor, ingresa el nombre del módulo.")

        st.markdown("---")
        st.markdown("#### **Lista de Módulos**")
        
        modules_data = []
        for modulo, info in st.session_state.escuelas[school_to_modify].items():
            modules_data.append({
                "Módulo": modulo,
                "Duración": info.get("duracion", "N/A"),
                "Estado": "Completado" if info.get("evaluado") else "Pendiente",
            })
        
        st.dataframe(modules_data, use_container_width=True)
        
        st.markdown("#### **Eliminar Módulo**")
        modulo_a_eliminar = st.selectbox("Selecciona un módulo para eliminar:", options=list(st.session_state.escuelas[school_to_modify].keys()))
        if st.button("🗑️ Eliminar Módulo"):
            if modulo_a_eliminar:
                del st.session_state.escuelas[school_to_modify][modulo_a_eliminar]
                st.success(f"Módulo '{modulo_a_eliminar}' eliminado de la escuela '{school_to_modify}'.")
                st.rerun()

with tab6:
    st.header("📊 Analíticas y Reportes")
    st.write("Una visión general del progreso de los usuarios y el rendimiento de los temas de capacitación.")

    # 1. Reporte de Progreso de Usuarios
    st.subheader("1. Progreso General de Usuarios")
    
    analytics_data = []
    for user, data in st.session_state.users.items():
        total_temas_completados = sum(len(temas) for temas in data["temas_completados"].values())
        total_puntaje = sum(sum(temas.values()) for temas in data["temas_completados"].values())
        promedio_final = total_puntaje / total_temas_completados if total_temas_completados > 0 else 0
        analytics_data.append({
            "Usuario": user,
            "Puntos": data.get("puntos", 0),
            "Nivel": data.get("nivel", 0),
            "Módulos Completados": total_temas_completados,
            "Puntaje Promedio": promedio_final,
        })
    
    df_analytics = pd.DataFrame(analytics_data)
    
    if not df_analytics.empty:
        st.dataframe(df_analytics, use_container_width=True)
    else:
        st.info("Aún no hay datos de usuarios para mostrar.")
    
    st.markdown("---")

    # 2. Rendimiento por Tema
    st.subheader("2. Rendimiento Promedio por Módulo")
    
    tema_scores = {}
    for user, data in st.session_state.users.items():
        for escuela, temas in data["temas_completados"].items():
            for tema, puntaje in temas.items():
                if tema not in tema_scores:
                    tema_scores[tema] = []
                tema_scores[tema].append(puntaje)

    if tema_scores:
        promedios_tema = {tema: sum(scores) / len(scores) for tema, scores in tema_scores.items()}
        df_promedios = pd.DataFrame(promedios_tema.items(), columns=["Módulo", "Puntaje Promedio"])
        st.bar_chart(df_promedios, x="Módulo", y="Puntaje Promedio")
        
        st.write("Este gráfico muestra el puntaje promedio de todos los usuarios por cada módulo completado. Las puntuaciones más bajas pueden indicar un tema que requiere más atención o una actualización de los materiales de capacitación.")
    else:
        st.info("Aún no hay módulos completados para generar el gráfico de rendimiento.")

    st.markdown("---")

    # 3. Puntaje Promedio por Escuela
    st.subheader("3. Puntaje Promedio por Escuela")
    school_scores = {}
    for school, topics in st.session_state.escuelas.items():
        total_score = 0
        evaluated_topics = 0
        for topic, data in topics.items():
            if data["evaluado"]:
                total_score += data["puntaje"]
                evaluated_topics += 1
        if evaluated_topics > 0:
            school_scores[school] = total_score / evaluated_topics
    
    if school_scores:
        df_school_scores = pd.DataFrame(school_scores.items(), columns=["Escuela", "Puntaje Promedio"])
        st.bar_chart(df_school_scores, x="Escuela", y="Puntaje Promedio")
        st.write("Este gráfico muestra la calificación promedio de los módulos evaluados por escuela. Es un indicador clave del rendimiento general de cada área de capacitación.")
    else:
        st.info("No hay suficientes datos para generar el gráfico de rendimiento por escuela.")

    st.markdown("---")

    # 4. Top 5 Temas Mejor Puntuados
    st.subheader("4. Top 5 Temas Mejor Puntuados")
    if tema_scores:
        promedios_tema = {tema: sum(scores) / len(scores) for tema, scores in tema_scores.items()}
        top_5_temas = sorted(promedios_tema.items(), key=lambda item: item[1], reverse=True)[:5]
        df_top_5 = pd.DataFrame(top_5_temas, columns=["Tema", "Puntaje Promedio"])
        st.table(df_top_5)
    else:
        st.info("Aún no hay módulos completados para mostrar el ranking.")

    st.markdown("---")
    
    # 5. Distribución de Puntos por Usuario
    st.subheader("5. Distribución de Puntos por Usuario")
    if not df_analytics.empty:
        df_analytics_points = df_analytics[["Usuario", "Puntos"]]
        st.bar_chart(df_analytics_points, x="Usuario", y="Puntos")
        st.write("Este gráfico muestra los puntos de gamificación de cada usuario, lo que permite visualizar a los usuarios más activos y comprometidos.")
    else:
        st.info("No hay datos de puntos de usuario para mostrar.")
