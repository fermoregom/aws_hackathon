import streamlit as st
import pandas as pd

# 👇 Esta línea debe ser la primera instrucción de Streamlit
st.set_page_config(page_title="AutoPartes AI", layout="wide")

# Cargar datos desde CSV local (puedes cambiarlo a S3 luego)
@st.cache_data
def cargar_catalogo():
    return pd.read_csv("base_autopartes_dummy.csv")

catalogo_df = cargar_catalogo()

# ----------------- UI Layout -----------------

# Sidebar para navegación
seccion = st.sidebar.radio("Navegar", ["Inicio", "Asistente AI", "Resultados", "Catálogo Completo"])

# ----------------- LANDING PAGE -----------------
if seccion == "Inicio":
    st.title("🔧 AutoPartes AI")
    st.subheader("Soluciones inteligentes para tus problemas automotrices.")
    st.markdown("""
    Bienvenido a nuestra plataforma demo, donde un agente inteligente te ayudará a encontrar la pieza ideal para tu automóvil.  
    **¿Tienes una pieza dañada o necesitas reemplazo?**  
    👉 Ve a la sección *Asistente AI* y empieza la conversación.
    """)
    st.image("https://cdn.pixabay.com/photo/2016/02/19/10/00/car-1209912_1280.jpg", use_column_width=True)

# ----------------- CHAT CON AGENTE -----------------
elif seccion == "Asistente AI":
    st.title("🤖 Asistente de AutoPartes")

    if "chat_historial" not in st.session_state:
        st.session_state.chat_historial = []

    # Mostrar historial
    for msg in st.session_state.chat_historial:
        st.chat_message(msg["role"]).write(msg["content"])

    # Entrada del usuario
    user_input = st.chat_input("Describe tu problema aquí...")
    if user_input:
        st.session_state.chat_historial.append({"role": "user", "content": user_input})

        # Aquí puedes conectar con tu agente o modelo LLM
        # Simulación de respuesta
        respuesta = "Gracias por la descripción. Parece que necesitas un *alternador* compatible con Nissan Altima 2015. Buscando..."
        
        st.session_state.chat_historial.append({"role": "assistant", "content": respuesta})
        st.chat_message("assistant").write(respuesta)

# ----------------- RESULTADOS -----------------
elif seccion == "Resultados":
    st.title("📋 Resultados del Análisis")
    st.markdown("Aquí mostraremos las piezas compatibles que encontró el agente.")

    # Simular que el agente identificó un modelo específico (esto debe venir del LLM en producción)
    filtro = catalogo_df[
        (catalogo_df["Marca de Auto"] == "Nissan") &
        (catalogo_df["Modelo"] == "Altima") &
        (catalogo_df["Nombre de Pieza"] == "Alternador")
    ]

    st.dataframe(filtro, use_container_width=True)

# ----------------- CATÁLOGO -----------------
elif seccion == "Catálogo Completo":
    st.title("📦 Catálogo Completo de Autopartes")
    st.dataframe(catalogo_df, use_container_width=True)
