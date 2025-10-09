import streamlit as st

st.set_page_config(page_title="Home", page_icon="📊", layout="wide")

# --- Sidebar global info ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=100)
    st.markdown("### 💡 *Modelos disponibles:*")
    st.markdown(
        "- 🧮 **Predicción de Notas (Clasificación)**\n- 🏠 **Precios de Viviendas (Regresión)**"
    )
    st.markdown("---")
    st.markdown("👨‍💻 **Juan Sebastián Sánchez Hincapié**")
    st.markdown("[🌐 LinkedIn](https://www.linkedin.com/in/jssanchezh/)")
    st.caption("Desarrollado con ❤️ en Python y Streamlit")

# --- Layout principal ---
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=120)
with col2:
    st.title("📊 Sistema de Predicción")
    st.write(
        "Bienvenido. Usa la barra lateral para navegar entre los modelos disponibles."
    )

st.divider()

st.header("🔍 ¿Qué puedes hacer aquí?")
st.write(
    """
Este sistema utiliza **modelos de Machine Learning** entrenados para predecir distintos tipos de valores,
como precios de vivienda o desempeño académico.

Desde la barra lateral puedes:
- 🧮 Ingresar datos manualmente
- 📂 Subir archivos CSV para predicciones masivas
- 📈 Visualizar resultados de forma inmediata
"""
)

st.divider()

st.info(
    "Proyecto académico desarrollado en Python con Streamlit. \
Permite explorar y realizar predicciones con modelos entrenados de clasificación y regresión."
)
