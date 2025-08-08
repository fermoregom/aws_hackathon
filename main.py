import streamlit as st
import pandas as pd

# Cargar catálogo desde CSV del proyecto
catalogo_df = pd.read_csv("base_autopartes_dummy.csv")

# Configuración inicial (debe ser la primera instrucción de Streamlit)
st.set_page_config(page_title="AutoPartes AI", layout="wide")



# Variable para controlar la pestaña activa
if "show_assistant" not in st.session_state:
    st.session_state.show_assistant = False

# Tabs en la parte superior en lugar de sidebar
tabs = st.tabs(["🏠 Inicio", "🤖 Asistente AI", "📋 Resultados", "📦 Catálogo Completo"])

# ---------------- INICIO ----------------
with tabs[0]:
    # Contenedor para imagen con botón superpuesto
    st.markdown(
        """
        <style>
        .image-container {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        .overlay-button {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 10;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Mostrar imagen
    st.image("AutoPartes AI_ Tradición y Tecnología.png", use_column_width=True)
    
    # Botón centrado
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔍 Buscar Pieza", key="buscar_pieza", use_container_width=True):
            st.session_state.show_assistant = True
            st.rerun()
    
    # Mostrar asistente si se presionó el botón
    if st.session_state.show_assistant:
        st.title("🤖 Asistente de AutoPartes")
        
        if "chat_historial" not in st.session_state:
            st.session_state.chat_historial = []
        
        for msg in st.session_state.chat_historial:
            st.chat_message(msg["role"]).write(msg["content"])
        
        user_input = st.chat_input("Describe tu problema aquí...")
        if user_input:
            st.session_state.chat_historial.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            respuesta = "Gracias por la descripción. Parece que necesitas un *alternador* compatible con Nissan Altima 2015. Buscando..."
            st.session_state.chat_historial.append({"role": "assistant", "content": respuesta})
            st.chat_message("assistant").write(respuesta)

# ---------------- ASISTENTE AI ----------------
with tabs[1]:
    if not st.session_state.show_assistant:
        st.title("🤖 Asistente de AutoPartes")
        
        if "chat_historial" not in st.session_state:
            st.session_state.chat_historial = []
        
        for msg in st.session_state.chat_historial:
            st.chat_message(msg["role"]).write(msg["content"])
        
        user_input = st.chat_input("Describe tu problema aquí...")
        if user_input:
            st.session_state.chat_historial.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            respuesta = "Gracias por la descripción. Parece que necesitas un *alternador* compatible con Nissan Altima 2015. Buscando..."
            st.session_state.chat_historial.append({"role": "assistant", "content": respuesta})
            st.chat_message("assistant").write(respuesta)
    else:
        st.info("El asistente ya está activo en la pestaña de Inicio. Haz clic allí para continuar.")

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
    
    # Filtros
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        marcas = st.multiselect("Marca de Auto", catalogo_df["Marca de Auto"].unique())
    with col2:
        modelos = st.multiselect("Modelo", catalogo_df["Modelo"].unique())
    with col3:
        piezas = st.multiselect("Nombre de Pieza", catalogo_df["Nombre de Pieza"].unique())
    with col4:
        precios = st.selectbox("Filtrar por precio", ["Todos", "Menor a $100", "$100-$200", "Mayor a $200"])
    
    # Aplicar filtros
    df_filtrado = catalogo_df.copy()
    
    if marcas:
        df_filtrado = df_filtrado[df_filtrado["Marca de Auto"].isin(marcas)]
    if modelos:
        df_filtrado = df_filtrado[df_filtrado["Modelo"].isin(modelos)]
    if piezas:
        df_filtrado = df_filtrado[df_filtrado["Nombre de Pieza"].isin(piezas)]
    
    st.dataframe(df_filtrado, use_container_width=True)
