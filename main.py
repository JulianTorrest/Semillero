import streamlit as st
import random
import spacy

# --- 1. Base de Conocimientos (con más detalle) ---
conocimiento_chatbot = {
    "cartera": {
        "palabras_clave": ["cartera", "clientes"],
        "respuesta": "La cartera de clientes es el activo más valioso de una empresa, compuesto por todos los clientes actuales y potenciales. Una gestión eficaz permite no solo retenerlos, sino también maximizar su valor a lo largo del tiempo."
    },
    "gestion de cartera": {
        "palabras_clave": ["gestión", "manejo", "administracion de cartera"],
        "respuesta": "La gestión de cartera implica una serie de estrategias para administrar la relación con los clientes. Esto incluye la segmentación, la personalización de la comunicación, la optimización de los ciclos de vida del cliente y la resolución proactiva de sus problemas."
    },
    "clientes conflictivos": {
        "palabras_clave": ["conflictivos", "dificiles", "queja", "insatisfaccion"],
        "respuesta": "Manejar a un cliente conflictivo requiere inteligencia emocional y profesionalidad. Las mejores prácticas incluyen escuchar activamente, validar sus sentimientos, no tomar la crítica como algo personal y, finalmente, ofrecer una solución justa y rápida para reconstruir la confianza."
    },
    "fidelizacion": {
        "palabras_clave": ["fidelizacion", "lealtad", "retencion", "retener clientes"],
        "respuesta": "La fidelización de clientes se centra en convertir a compradores ocasionales en embajadores de la marca. Esto se logra a través de un servicio excepcional, programas de lealtad, comunicación personalizada y un seguimiento post-venta que demuestre que su relación es valorada."
    },
    "cierre de ventas": {
        "palabras_clave": ["cierre de ventas", "cerrar trato", "negociacion"],
        "respuesta": "El cierre de ventas es el punto culminante del ciclo de ventas. No se trata solo de conseguir la firma, sino de asegurar que el cliente se sienta seguro y satisfecho con su decisión. Técnicas como el 'cierre por asunción' o el 'cierre de la pregunta' son muy útiles para guiar al cliente a la acción."
    }
}

examenes = {
    "dificultad_baja": [
        {
            "tipo": "opcion_multiple",
            "pregunta": "¿Cuál es el principal objetivo de la gestión de cartera?",
            "opciones": ["A) Vender nuevos productos.", "B) Maximizar el valor de los clientes a largo plazo.", "C) Reducir costos de marketing."],
            "respuesta_correcta": "B",
            "explicacion": "El objetivo principal no es solo vender, sino construir relaciones duraderas que maximicen el valor total que un cliente aporta a la empresa."
        },
        {
            "tipo": "opcion_multiple",
            "pregunta": "¿Qué es la fidelización de clientes?",
            "opciones": ["A) Un programa de descuentos.", "B) El conjunto de técnicas para lograr la lealtad de un cliente.", "C) La venta de nuevos productos a clientes existentes."],
            "respuesta_correcta": "B",
            "explicacion": "La fidelización es una estrategia a largo plazo para construir una relación sólida que incentive la lealtad del cliente."
        }
    ],
    "dificultad_media": [
        {
            "tipo": "opcion_multiple",
            "pregunta": "¿Qué estrategia es efectiva para un cliente que expresa insatisfacción?",
            "opciones": ["A) Ignorar su queja.", "B) Escuchar activamente y mostrar empatía.", "C) Ofrecerle un descuento sin escucharlo."],
            "respuesta_correcta": "B",
            "explicacion": "La empatía y la escucha activa son el primer paso para desactivar la situación y encontrar una solución adecuada."
        },
        {
            "tipo": "texto_libre",
            "pregunta": "Describe brevemente cómo la gestión de cartera puede ayudar a una empresa a crecer."
        }
    ],
    "dificultad_alta": [
        {
            "tipo": "opcion_multiple",
            "pregunta": "¿Cuál es la principal diferencia entre un cliente satisfecho y uno leal?",
            "opciones": ["A) El cliente leal compra más a menudo.", "B) El cliente leal no es sensible al precio y recomienda la marca, a diferencia del satisfecho.", "C) No hay diferencia, son el mismo concepto."],
            "respuesta_correcta": "B",
            "explicacion": "La lealtad va más allá de la satisfacción. Un cliente leal es un promotor de la marca y es más resistente a las ofertas de la competencia."
        },
        {
            "tipo": "texto_libre",
            "pregunta": "Explica la importancia de la 'segmentación de clientes' en la gestión de cartera y cómo se puede aplicar."
        }
    ]
}

# --- 2. Configuración de la Aplicación y del Estado de la Sesión ---
st.set_page_config(page_title="Asistente de Gestión de Cartera", layout="wide", initial_sidebar_state="expanded")

# Inicializar spacy una sola vez
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("es_core_news_sm")
        return nlp
    except OSError:
        st.error("El modelo de spaCy 'es_core_news_sm' no está instalado. Por favor, ejecuta 'python -m spacy download es_core_news_sm' en tu terminal.")
        st.stop()
nlp = load_spacy_model()

if 'modo' not in st.session_state:
    st.session_state.modo = 'inicio'
if 'puntuacion' not in st.session_state:
    st.session_state.puntuacion = 0
if 'pregunta_actual' not in st.session_state:
    st.session_state.pregunta_actual = 0
if 'preguntas_examen' not in st.session_state:
    st.session_state.preguntas_examen = []
if 'respuestas_usuario' not in st.session_state:
    st.session_state.respuestas_usuario = {}

# --- 3. Funciones de Lógica de la Aplicación ---
def cambiar_modo(nuevo_modo):
    st.session_state.modo = nuevo_modo
    st.session_state.puntuacion = 0
    st.session_state.pregunta_actual = 0
    st.session_state.preguntas_examen = []
    st.session_state.respuestas_usuario = {}

def iniciar_examen(dificultad):
    preguntas_seleccionadas = examenes.get(dificultad, [])
    random.shuffle(preguntas_seleccionadas)
    st.session_state.preguntas_examen = preguntas_seleccionadas
    cambiar_modo('examen_en_curso')

def procesar_respuesta_opcion_multiple(respuesta_usuario, pregunta_info):
    st.session_state.respuestas_usuario[st.session_state.pregunta_actual] = respuesta_usuario
    if respuesta_usuario == pregunta_info["respuesta_correcta"]:
        st.session_state.puntuacion += 1
        st.success("¡Respuesta correcta! 🎉")
    else:
        st.error(f"Respuesta incorrecta. La respuesta correcta es {pregunta_info['respuesta_correcta']}.")
        with st.expander("Ver explicación"):
            st.info(pregunta_info["explicacion"])
    st.session_state.pregunta_actual += 1

def procesar_respuesta_texto_libre(respuesta_usuario):
    st.session_state.respuestas_usuario[st.session_state.pregunta_actual] = respuesta_usuario
    st.info("Gracias por tu respuesta. Tu respuesta será revisada por el evaluador.")
    st.session_state.pregunta_actual += 1
    
# --- 4. Interfaz de Usuario Principal (UI) ---
st.title("Asistente de Gestión de Cartera y Clientes")
st.markdown("---")

# Menú principal en la barra lateral
st.sidebar.header("Menú Principal")
if st.sidebar.button("🏠 Inicio"):
    cambiar_modo('inicio')
if st.sidebar.button("🗣️ Chatbot"):
    cambiar_modo('chatbot')
if st.sidebar.button("✍️ Examen"):
    cambiar_modo('seleccion_examen')

if st.session_state.modo == 'inicio':
    st.header("Bienvenido a tu Asistente de Formación")
    st.write("Selecciona una opción del menú para comenzar:")
    st.info("🗣️ **Chatbot**: Responde preguntas sobre gestión de cartera y clientes.")
    st.success("✍️ **Examen**: Evalúa tus conocimientos con pruebas de diferentes dificultades.")

elif st.session_state.modo == 'chatbot':
    st.header("🗣️ Modo Chatbot: Preguntas Libres")
    st.write("Haz una pregunta sobre **gestión de cartera, clientes conflictivos, fidelización** y más.")
    pregunta_usuario = st.text_input("Tu pregunta:", key="chat_input")
    
    if pregunta_usuario:
        doc = nlp(pregunta_usuario.lower())
        respuesta_encontrada = False
        for tema, info in conocimiento_chatbot.items():
            for palabra in info["palabras_clave"]:
                if doc.similarity(nlp(palabra)) > 0.6:  # Usa similitud de spaCy
                    st.info(info["respuesta"])
                    respuesta_encontrada = True
                    break
            if respuesta_encontrada:
                break
        
        if not respuesta_encontrada:
            st.warning("Lo siento, no tengo una respuesta detallada para eso. Intenta con un tema más específico como 'fidelización' o 'cierre de ventas'.")

elif st.session_state.modo == 'seleccion_examen':
    st.header("✍️ Selecciona la Dificultad del Examen")
    st.markdown("Elige el nivel para comenzar tu evaluación.")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Dificultad Baja 🟢", use_container_width=True):
            iniciar_examen("dificultad_baja")
    with col2:
        if st.button("Dificultad Media 🟡", use_container_width=True):
            iniciar_examen("dificultad_media")
    with col3:
        if st.button("Dificultad Alta 🔴", use_container_width=True):
            iniciar_examen("dificultad_alta")

elif st.session_state.modo == 'examen_en_curso':
    st.header("✍️ Examen de Conocimientos")
    
    if st.session_state.pregunta_actual < len(st.session_state.preguntas_examen):
        pregunta = st.session_state.preguntas_examen[st.session_state.pregunta_actual]
        st.subheader(f"Pregunta {st.session_state.pregunta_actual + 1}: {pregunta['pregunta']}")

        if pregunta["tipo"] == "opcion_multiple":
            opciones_elegidas = st.radio("Elige una opción:", pregunta["opciones"], key=f"radio_{st.session_state.pregunta_actual}")
            if st.button("Responder", key=f"btn_responder_{st.session_state.pregunta_actual}"):
                opcion_elegida = opciones_elegidas.split(')')[0]
                procesar_respuesta_opcion_multiple(opcion_elegida, pregunta)
        
        elif pregunta["tipo"] == "texto_libre":
            respuesta_usuario = st.text_area("Escribe tu respuesta aquí:", key=f"text_area_{st.session_state.pregunta_actual}", height=200)
            if st.button("Enviar Respuesta", key=f"btn_enviar_{st.session_state.pregunta_actual}"):
                procesar_respuesta_texto_libre(respuesta_usuario)

    else:
        st.header("✅ Examen Finalizado")
        puntuacion_final = st.session_state.puntuacion
        preguntas_correctas = len([p for p in st.session_state.respuestas_usuario.values() if p in "ABC"])
        total_preguntas_multiples = len([p for p in st.session_state.preguntas_examen if p["tipo"] == "opcion_multiple"])

        st.metric(label="Tu puntuación final (Preguntas de Opción Múltiple)", value=f"{puntuacion_final} de {total_preguntas_multiples}")
        st.balloons()

        st.subheader("Revisión del Examen")
        with st.expander("Ver respuestas incorrectas"):
            for i, pregunta_info in enumerate(st.session_state.preguntas_examen):
                if pregunta_info["tipo"] == "opcion_multiple":
                    respuesta_correcta = pregunta_info["respuesta_correcta"]
                    respuesta_dada = st.session_state.respuestas_usuario.get(i)
                    if respuesta_dada != respuesta_correcta:
                        st.error(f"Pregunta {i + 1}: {pregunta_info['pregunta']}")
                        st.write(f"Tu respuesta: **{respuesta_dada}**")
                        st.write(f"Respuesta correcta: **{respuesta_correcta}**")
                        st.info(f"Explicación: {pregunta_info['explicacion']}")
                elif pregunta_info["tipo"] == "texto_libre":
                    respuesta_dada = st.session_state.respuestas_usuario.get(i)
                    st.warning(f"Pregunta {i + 1}: {pregunta_info['pregunta']}")
                    st.write("Tu respuesta:")
                    st.markdown(f"```\n{respuesta_dada}\n```")

        if st.button("Volver a iniciar examen"):
            cambiar_modo('seleccion_examen')
