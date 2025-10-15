import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(page_title="Predicción de Avalúo de Viviendas", page_icon="🏡")

st.title("🏡 Predicción del Avalúo de Propiedades")
st.markdown(
    """
    Esta aplicación permite estimar el **avalúo total de un inmueble**
    a partir de sus características físicas y de entorno.
    """
)


# =========================================================
# CARGA DE MODELOS Y RECURSOS
# =========================================================
@st.cache_resource
def load_resources():
    base_path = os.path.dirname(__file__)

    model_path = os.path.join(
        base_path, "..", "models", "precios_casas_cali", "br_model.pkl"
    )
    scaler_path = os.path.join(
        base_path, "..", "models", "precios_casas_cali", "standard_scaler.pkl"
    )
    encoder_path = os.path.join(
        base_path, "..", "models", "precios_casas_cali", "onehotencoder.pkl"
    )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    return model, scaler, encoder


model, scaler, encoder = load_resources()

# =========================================================
# CONFIGURACIÓN DE VARIABLES
# =========================================================
cols_numericas = [
    "AREA_CONSTRUCCION",  # nombre corregido según el modelo
    "AREA_TERRENO",
    "ANTIGUEDAD",
    "ESTRATO",
    "AREA_ANEXO",
]
cols_categoricas = ["BARRIO", "TIENE_ANEXO"]
todas_cols = cols_numericas + cols_categoricas

modo = st.radio(
    "Selecciona el modo de ingreso de datos:", ["Ingreso manual", "Subir archivo CSV"]
)


# =========================================================
# FUNCIÓN AUXILIAR DE PREPROCESAMIENTO
# =========================================================
def preparar_datos(df):
    """Ajusta nombres, genera TIENE_ANEXO y transforma datos."""
    # Renombrar columnas si es necesario
    df = df.rename(columns={"AREA_CONSTRUIDA": "AREA_CONSTRUCCION"})

    # Crear la columna TIENE_ANEXO automáticamente
    if "TIENE_ANEXO" not in df.columns:
        df["TIENE_ANEXO"] = (df["AREA_ANEXO"] > 0).astype(int)

    # Escalar variables numéricas
    scaled = scaler.transform(df[cols_numericas])
    df_scaled = pd.DataFrame(scaled, columns=cols_numericas)

    # Codificar variables categóricas
    encoded = encoder.transform(df[cols_categoricas])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(cols_categoricas)
    )

    # Concatenar ambas partes
    X = pd.concat([df_scaled, encoded_df], axis=1)
    return X


# =========================================================
# MODO 1 — INGRESO MANUAL
# =========================================================
if modo == "Ingreso manual":
    st.subheader("✍️ Ingreso manual de características")

    col1, col2 = st.columns(2)

    with col1:
        area_const = st.number_input(
            "Área construida (m²)", min_value=10.0, value=80.0, step=1.0
        )
        area_terr = st.number_input(
            "Área del terreno (m²)", min_value=20.0, value=120.0, step=1.0
        )
        estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])
        antiguedad = st.number_input(
            "Antigüedad (años)", min_value=0, max_value=100, value=10
        )

    with col2:
        area_anexo = st.number_input(
            "Área del anexo (m²)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Si no tiene anexo, dejar en 0.",
        )
        barrio = st.text_input("Barrio o zona", value="Centro")

    df_input = pd.DataFrame(
        [[area_const, area_terr, antiguedad, estrato, area_anexo, barrio]],
        columns=[
            "AREA_CONSTRUCCION",
            "AREA_TERRENO",
            "ANTIGUEDAD",
            "ESTRATO",
            "AREA_ANEXO",
            "BARRIO",
        ],
    )

    df_input["TIENE_ANEXO"] = (df_input["AREA_ANEXO"] > 0).astype(int)

    if st.checkbox("📋 Mostrar datos ingresados"):
        st.dataframe(df_input)

    if st.button("🔮 Predecir Avalúo"):
        try:
            X = preparar_datos(df_input)
            prediccion = model.predict(X)[0]
            st.success(f"💰 Avalúo estimado del inmueble: **${prediccion:,.0f} COP**")

        except Exception as e:
            st.error(f"⚠️ Error durante la predicción: {e}")
            st.write(
                "Columnas esperadas:",
                getattr(model, "feature_names_in_", "No disponible"),
            )

# =========================================================
# MODO 2 — ARCHIVO CSV
# =========================================================
else:
    st.subheader("📂 Predicción masiva mediante archivo CSV")
    st.markdown(
        """
        Sube un archivo `.csv` que contenga las siguientes columnas:
        - **AREA_CONSTRUCCION** (o AREA_CONSTRUIDA, se ajusta automáticamente)
        - **AREA_TERRENO**
        - **ANTIGUEDAD**
        - **ESTRATO**
        - **AREA_ANEXO**
        - **BARRIO**
        """
    )

    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.write("Vista previa de los datos:")
            st.dataframe(df_input.head())

            # Ajustar nombres y generar columnas necesarias
            df_input = df_input.rename(columns={"AREA_CONSTRUIDA": "AREA_CONSTRUCCION"})
            if "TIENE_ANEXO" not in df_input.columns:
                df_input["TIENE_ANEXO"] = (df_input["AREA_ANEXO"] > 0).astype(int)

            # Validar columnas
            missing_cols = [c for c in todas_cols if c not in df_input.columns]
            if missing_cols:
                st.error(f"❌ Faltan columnas obligatorias: {missing_cols}")
            else:
                X = preparar_datos(df_input)
                predicciones = model.predict(X)
                df_input["AVALUO_PREDICHO"] = predicciones

                st.success("✅ Predicciones generadas correctamente.")
                st.dataframe(df_input.head())

                csv_output = df_input.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar archivo con predicciones",
                    data=csv_output,
                    file_name="predicciones_avaluos.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"⚠️ Error al procesar el archivo: {e}")
