import streamlit as st
import pandas as pd
import joblib
import os

# --- Cargar modelo y scaler ---
@st.cache_resource
def load_resources():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, '..', 'models','precios_casas_boston', 'br_model.pkl')
    scaler_path = os.path.join(base_path, '..', 'models','precios_casas_boston', 'standard_scaler.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_resources()

# --- Título ---
st.title("Predicción del Valor de la Vivienda")
st.write("Ingrese los valores para predecir el valor promedio de la vivienda (`medv`).")

# --- Entradas del usuario ---
crim = st.number_input("crim: Tasa de crimen per cápita", min_value=0.0, value=0.01, step=0.01)
zn = st.number_input("zn: Proporción de terrenos residenciales", min_value=0.0, value=0.0)
indus = st.number_input("indus: Áreas industriales (acres)", min_value=0.0, value=2.0)
chas = st.selectbox("chas: ¿Frente al río Charles?", options=[0, 1])
nox = st.number_input("nox: Contaminación por óxidos nítricos", min_value=0.0, value=0.5, step=0.01)
rm = st.number_input("rm: Número promedio de habitaciones", min_value=1.0, value=6.0)
age = st.number_input("age: % de unidades antiguas (>1940)", min_value=0.0, value=60.0)
dis = st.number_input("dis: Distancia ponderada a centros de empleo", min_value=0.0, value=4.0)
rad = st.number_input("rad: Índice de accesibilidad a autopistas", min_value=1, max_value=24, value=1, step=1)
tax = st.number_input("tax: Tasa de impuesto a la propiedad", min_value=100, max_value=800, value=300, step=1)
ptratio = st.number_input("ptratio: Relación alumnos/profesor", min_value=10.0, max_value=30.0, value=15.0)
black = st.number_input("black: 1000(Bk - 0.63)^2", min_value=0.0, value=390.0)
lstat = st.number_input("lstat: % de personas con estatus bajo", min_value=0.0, value=12.0)

# --- Crear DataFrame de entrada ---
input_data = pd.DataFrame([[
    crim, zn, indus, chas, nox, rm, age, dis,
    rad, tax, ptratio, black, lstat
]], columns=[
    'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
    'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat'
])

# --- Mostrar input si el usuario lo desea ---
if st.checkbox("Mostrar datos de entrada"):
    st.write("Datos ingresados:")
    st.dataframe(input_data)

# --- Predicción ---
if st.button("Predecir valor de vivienda"):
    try:
        # Escalar
        input_scaled = scaler.transform(input_data)

        input_array = input_scaled.reshape(1, -1)

        # Predecir
        prediction = model.predict(input_array)[0]
        st.subheader(f"Predicción del valor promedio de la vivienda: **${prediction:.2f} mil dólares**")

    except Exception as e:
        st.write("Columnas esperadas por el modelo:", getattr(model, "feature_names_in_", "No disponible"))
        st.write("Columnas input:", input_data.columns.tolist())
        st.error(f"Ocurrió un error durante la predicción: {e}")
        st.write("Datos procesados:", input_data)
