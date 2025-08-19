import streamlit as st
import random

# --- 1. Base de Conocimientos ---
# Aquí se almacenan las preguntas y respuestas para el chatbot y los exámenes.

# Para preguntas y respuestas del chatbot (modo de texto libre)
conocimiento_chatbot = {
    "cartera": "La cartera de clientes es el conjunto de clientes activos y potenciales que una empresa ha cultivado a lo largo del tiempo. Es el activo más valioso de una organización.",
    "gestion de cartera": "La gestión de cartera se refiere a las estrategias y procesos para administrar eficientemente las relaciones con los clientes, maximizando su valor a largo plazo y su fidelidad. Incluye la segmentación de clientes, la personalización de la comunicación y la resolución de problemas.",
    "clientes conflictivos": "Para manejar clientes conflictivos, es fundamental mantener la calma y una actitud profesional. Las mejores estrategias son: escuchar activamente la queja del cliente, mostrar empatía, no tomar la crítica como algo personal y proponer soluciones concretas y rápidas. El objetivo es convertir una mala experiencia en una oportunidad para fortalecer la relación.",
    "fidelizacion": "La fidelización de clientes es el conjunto de técnicas para lograr que un cliente regular se convierta en un cliente leal a largo plazo. Las tácticas incluyen programas de recompensas, excelente servicio al cliente, comunicación personalizada y el seguimiento post-venta.",
    "cierre de ventas": "El cierre de ventas es la etapa final del proceso de ventas, donde el vendedor guía al cliente a tomar la decisión de compra. Hay varias técnicas de cierre, como el 'cierre por asunción', donde se asume que el cliente ya decidió comprar y se avanza a los detalles de pago."
}

# Para los exámenes de evaluación (opción múltiple)
examenes = {
    "dificultad_baja": [
        {
            "pregunta": "¿Qué es la fidelización de clientes?",
            "opciones": ["A) Aumentar los precios de los productos.", "B) El conjunto de técnicas para lograr la lealtad de un cliente.", "C) La creación de nuevos productos."],
            "respuesta_correcta": "B"
        },
        {
            "pregunta": "¿Cuál es un beneficio clave de una buena gestión de cartera?",
            "opciones": ["A) Aumenta la rotación del personal.", "B) Incrementa la fidelidad del cliente.", "C) Reduce el número de productos."],
            "respuesta_correcta": "B"
        }
    ],
    "dificultad_media": [
        {
            "pregunta": "¿Qué estrategia es efectiva para un cliente que expresa insatisfacción?",
            "opciones": ["A) Ignorar su queja.", "B) Ofrecerle un descuento inmediatamente.", "C) Escuchar activamente y mostrar empatía."],
            "respuesta_correcta": "C"
        },
        {
            "pregunta": "¿Qué significa el 'cierre de ventas por asunción'?",
            "opciones": ["A) Dar por terminada la conversación.", "B) Asumir que el cliente ya compró y continuar con los siguientes pasos.", "C) Asumir la deuda del cliente."],
            "respuesta_correcta": "B"
        }
    ],
    "dificultad_alta": [
        {
            "pregunta": "¿Cuál es la principal diferencia entre un cliente satisfecho y uno leal?",
            "opciones": ["A) El cliente leal compra más a menudo.", "B) El cliente satisfecho vuelve a comprar, pero el cliente leal también recomienda la marca y es menos sensible al precio.", "C) No hay diferencia, son lo mismo."],
            "respuesta_correcta": "B"
        }
    ]
}

# --- 2. Configuración Inicial de la Aplicación y del Estado de la Sesión ---
st.set_page_config(page_title="Gestión de Cartera IA", layout="wide")

# Inicializar variables de estado para mantener la información entre interacciones del usuario
if 'modo' not in st.session_state:
    st.session_state.modo = 'chatbot'
if 'puntuacion' not in st.session_state:
    st.session_state.puntuacion = 0
if 'pregunta_actual' not in st.session_state:
    st.session_state.pregunta_actual = 0
if 'preguntas_examen' not in st.session_state:
    st.session_state.preguntas_examen = []

# --- 3. Funciones de Lógica de la Aplicación ---

def cambiar_modo(nuevo_modo):
    """Cambia el modo de la aplicación y reinicia el estado."""
    st.session_state.modo = nuevo_modo
    st.session_state.puntuacion = 0
    st.session_state.pregunta_actual = 0

def iniciar_examen(dificultad):
    """Selecciona las preguntas del examen y cambia al modo de examen."""
    preguntas_seleccionadas = examenes.get(dificultad, [])
    # Mezcla las preguntas para que el orden cambie en cada intento
    random.shuffle(preguntas_seleccionadas)
    st.session_state.preguntas_examen = preguntas_seleccionadas
    cambiar_modo('examen_en_curso')

def procesar_respuesta_examen(respuesta_usuario, respuesta_correcta):
    """Compara la respuesta del usuario con la correcta y actualiza la puntuación."""
    if respuesta_usuario == respuesta_correcta:
        st.session_state.puntuacion += 1
        st.success("¡Respuesta correcta! 🎉")
    else:
        st.error(f"Respuesta incorrecta. La respuesta correcta es {respuesta_correcta}.")
    
    # Avanza a la siguiente pregunta
    st.session_state.pregunta_actual += 1
    # Vuelve a ejecutar la página para mostrar la siguiente pregunta
    st.experimental_rerun()

# --- 4. Interfaz de Usuario Principal (UI) ---
st.title("Asistente de Gestión de Cartera y Clientes")
st.markdown("---")

# Barra lateral para navegar entre modos
st.sidebar.header("Menú Principal")
if st.sidebar.button("Chatbot (Preguntas Libres)"):
    cambiar_modo('chatbot')
if st.sidebar.button("Realizar Examen"):
    cambiar_modo('seleccion_examen')

# Lógica principal basada en el estado de la sesión
if st.session_state.modo == 'chatbot':
    st.header("Modo Chatbot")
    st.write("Escribe una pregunta sobre temas de cartera, clientes o gestión.")
    pregunta_usuario = st.text_input("Tu pregunta:", key="chat_input")
    
    if pregunta_usuario:
        respuesta = "Lo siento, no tengo información sobre ese tema. Intenta con 'cartera', 'clientes conflictivos' o 'fidelización'."
        
        # Búsqueda simple de palabras clave para encontrar la respuesta adecuada
        for palabra_clave, resp_predefinida in conocimiento_chatbot.items():
            if palabra_clave in pregunta_usuario.lower():
                respuesta = resp_predefinida
                break
        
        st.info(respuesta)

elif st.session_state.modo == 'seleccion_examen':
    st.header("Selecciona la Dificultad del Examen")
    col1, col2, col3 = st.columns(3)
    if col1.button("Dificultad Baja"):
        iniciar_examen("dificultad_baja")
    if col2.button("Dificultad Media"):
        iniciar_examen("dificultad_media")
    if col3.button("Dificultad Alta"):
        iniciar_examen("dificultad_alta")

elif st.session_state.modo == 'examen_en_curso':
    st.header("Examen de Conocimientos")
    
    # Mostrar la pregunta actual si aún hay preguntas
    if st.session_state.pregunta_actual < len(st.session_state.preguntas_examen):
        pregunta = st.session_state.preguntas_examen[st.session_state.pregunta_actual]
        
        st.subheader(f"Pregunta {st.session_state.pregunta_actual + 1}:")
        st.write(pregunta["pregunta"])
        
        # Mostrar las opciones de respuesta como radio buttons
        opciones_elegidas = st.radio("Elige una opción:", pregunta["opciones"], key=f"radio_{st.session_state.pregunta_actual}")
        
        # Botón para enviar la respuesta
        if st.button("Responder", key=f"btn_responder_{st.session_state.pregunta_actual}"):
            # Extrae solo la letra de la opción elegida (A, B, C)
            opcion_elegida = opciones_elegidas.split(')')[0]
            procesar_respuesta_examen(opcion_elegida, pregunta["respuesta_correcta"])

    else:
        # Examen finalizado
        st.header("Examen Finalizado")
        puntuacion_final = st.session_state.puntuacion
        total_preguntas = len(st.session_state.preguntas_examen)
        
        st.metric(label="Tu puntuación final", value=f"{puntuacion_final} de {total_preguntas}")
        st.balloons()
        
        if st.button("Volver a iniciar examen"):
            cambiar_modo('seleccion_examen')
