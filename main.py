import streamlit as st
import pandas as pd

# Simulación de catálogo (para que funcione sin error)
catalogo_df = pd.DataFrame({
    "Marca de Auto": ["Nissan", "Toyota"],
    "Modelo": ["Altima", "Corolla"],
    "Nombre de Pieza": ["Alternador", "Batería"],
    "Precio": ["$120", "$80"]
})

# Configuración inicial (debe ser la primera instrucción de Streamlit)
st.set_page_config(page_title="AutoPartes AI", layout="wide")



# Tabs en la parte superior en lugar de sidebar
tabs = st.tabs(["🏠 Inicio", "🤖 Asistente AI", "📋 Resultados", "📦 Catálogo Completo"])

# ---------------- INICIO ----------------
with tabs[0]:
    st.image("AutoPartes AI_ Tradición y Tecnología.png", use_column_width=True)

# ---------------- ASISTENTE AI ----------------
with tabs[1]:
    st.title("🤖 Asistente de AutoPartes")

    if "chat_historial" not in st.session_state:
        st.session_state.chat_historial = []

    for msg in st.session_state.chat_historial:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Describe tu problema aquí...")
    if user_input:
        st.session_state.chat_historial.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Simulación de respuesta
        respuesta = "Gracias por la descripción. Parece que necesitas un *alternador* compatible con Nissan Altima 2015. Buscando..."
        st.session_state.chat_historial.append({"role": "assistant", "content": respuesta})
        st.chat_message("assistant").write(respuesta)

# ---------------- RESULTADOS ----------------
with tabs[2]:
    st.title("📋 Resultados del Análisis")
    st.markdown("Aquí mostraremos las piezas compatibles que encontró el agente.")

    filtro = catalogo_df[
        (catalogo_df["Marca de Auto"] == "Nissan") &
        (catalogo_df["Modelo"] == "Altima") &
        (catalogo_df["Nombre de Pieza"] == "Alternador")
    ]

    st.dataframe(filtro, use_container_width=True)

# ---------------- CATÁLOGO COMPLETO ----------------
with tabs[3]:
    st.title("📦 Catálogo Completo de Autopartes")
    st.dataframe(catalogo_df, use_container_width=True)
