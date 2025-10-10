import streamlit as st
import pandas as pd
import joblib
import os
from io import StringIO

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Predicción Boston", page_icon="🏠")


# --- CARGA DE MODELOS ---
@st.cache_resource
def load_resources():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(
        base_path, "..", "models", "precios_casas_boston", "br_model.pkl"
    )
    scaler_path = os.path.join(
        base_path, "..", "models", "precios_casas_boston", "standard_scaler.pkl"
    )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


model, scaler = load_resources()

# --- TÍTULO ---
st.title("🏠 Predicción del Valor de Vivienda (Boston Housing Dataset)")
st.markdown("Selecciona el modo de entrada para realizar la predicción:")

# --- MODO DE ENTRADA ---
modo = st.radio("Modo de ingreso de datos:", ["Ingreso manual", "Subir archivo CSV"])

# --- VARIABLES ---
cols_to_scale = [
    "crim",
    "zn",
    "indus",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "black",
    "lstat",
]
all_columns = cols_to_scale + ["chas"]

# =========================================================
# MODO 1 — Ingreso manual
# =========================================================
if modo == "Ingreso manual":
    st.subheader("🔢 Ingreso manual de datos")

    col1, col2 = st.columns(2)

    with col1:
        crim = st.number_input(
            "crim: Tasa de crimen per cápita",
            min_value=0.0,
            value=0.01,
            step=0.00001,
            format="%.5f",
        )
        zn = st.number_input(
            "zn: % de terrenos residenciales", min_value=0.0, value=0.0
        )
        indus = st.number_input(
            "indus: Áreas industriales (acres)", min_value=0.0, value=2.0
        )
        chas = st.selectbox("chas: ¿Frente al río Charles?", options=[0, 1])
        nox = st.number_input(
            "nox: Contaminación por óxidos nítricos",
            min_value=0.0,
            value=0.5,
            step=0.00001,
            format="%.5f",
        )
        rm = st.number_input(
            "rm: Número promedio de habitaciones", min_value=1.0, value=6.0, step=0.1
        )

    with col2:
        age = st.number_input(
            "age: % de unidades antiguas (>1940)", min_value=0.0, value=60.0
        )
        dis = st.number_input(
            "dis: Distancia ponderada a centros de empleo", min_value=0.0, value=4.0
        )
        rad = st.number_input(
            "rad: Índice de accesibilidad a autopistas",
            min_value=1,
            max_value=24,
            value=1,
            step=1,
        )
        tax = st.number_input(
            "tax: Tasa de impuesto a la propiedad",
            min_value=100,
            max_value=1000,
            value=300,
            step=1,
        )
        ptratio = st.number_input(
            "ptratio: Relación alumnos/profesor",
            min_value=10.0,
            max_value=40.0,
            value=15.0,
        )
        black = st.number_input("black: 1000(Bk - 0.63)^2", min_value=0.0, value=390.0)
        lstat = st.number_input(
            "lstat: % de personas con estatus bajo", min_value=0.0, value=12.0
        )

    input_data = pd.DataFrame(
        [[crim, zn, indus, nox, rm, age, dis, rad, tax, ptratio, black, lstat, chas]],
        columns=cols_to_scale + ["chas"],
    )

    # --- Mostrar datos si el usuario desea ---
    if st.checkbox("Mostrar datos ingresados"):
        st.dataframe(input_data)

    # --- Predicción ---
    if st.button("🔮 Predecir valor de vivienda"):
        try:
            # Escalar solo las columnas numéricas
            input_scaled_values = scaler.transform(input_data[cols_to_scale])

            # Reconstruir el DataFrame escalado + chas
            input_scaled = pd.DataFrame(input_scaled_values, columns=cols_to_scale)
            input_scaled["chas"] = input_data["chas"].astype(float)

            # Predicción
            prediction = model.predict(input_scaled)[0]
            st.success(
                f"🏡 Valor estimado promedio de la vivienda: **${prediction:.2f} mil dólares**"
            )

        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
            st.write(
                "Columnas esperadas:",
                getattr(model, "feature_names_in_", "No disponible"),
            )
            st.write("Columnas input:", input_data.columns.tolist())

# =========================================================
# MODO 2 — Carga CSV
# =========================================================
else:
    st.subheader("📂 Predicción por archivo CSV")

    st.markdown(
        """
    - Sube un archivo `.csv` que contenga las columnas:
      **crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat**
    - El sistema escalará los datos y devolverá una columna con las predicciones.
    """
    )

    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.write("Vista previa de tus datos:")
            st.dataframe(df_input.head())

            # Validar columnas
            missing_cols = [c for c in all_columns if c not in df_input.columns]
            if missing_cols:
                st.error(f"❌ Faltan las siguientes columnas: {missing_cols}")
            else:
                # Escalar solo las columnas numéricas
                scaled_values = scaler.transform(df_input[cols_to_scale])
                df_scaled = pd.DataFrame(scaled_values, columns=cols_to_scale)
                df_scaled["chas"] = df_input["chas"].astype(float)

                # Predicciones
                predictions = model.predict(df_scaled)
                df_input["prediccion_medv"] = predictions

                st.success("✅ Predicciones generadas correctamente.")
                st.dataframe(df_input.head())

                # Botón para descargar
                csv_output = df_input.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar resultados CSV",
                    data=csv_output,
                    file_name="predicciones_boston.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")
