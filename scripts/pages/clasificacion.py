import streamlit as st
import pandas as pd
import joblib
import os

# --- Cargar recursos (codificadores y modelo) ---
@st.cache_resource
def load_resources():
    base_path = os.path.dirname(__file__)
    
    onehot_path = os.path.join(base_path, '..', 'models', 'clasificacion', 'onehotencoder.pkl')
    scaler_path = os.path.join(base_path, '..', 'models', 'clasificacion', 'standard_scaler.pkl')
    model_path = os.path.join(base_path, '..', 'models', 'clasificacion', 'rf_model.pkl')

    onehot_encoder = joblib.load(onehot_path)
    standard_scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return onehot_encoder, standard_scaler, model

onehot_encoder, standard_scaler, model = load_resources()

# --- Título e instrucciones ---
st.title('Predicción de Aprobación del Curso')
st.write('Ingrese la información del estudiante para predecir si aprobará el curso.')

# --- Entradas del usuario ---
felder_options = ['activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal']
felder = st.selectbox('Felder', felder_options)
examen_admision = st.number_input('Examen de admisión Universidad', min_value=0.0, max_value=5.0, step=0.01)

# --- Crear DataFrame con los datos ingresados ---
input_data = pd.DataFrame([[felder, examen_admision]], columns=['Felder', 'Examen_admisión_Universidad'])

# --- Preprocesamiento ---
try:
    # Codificar 'Felder'
    felder_encoded = onehot_encoder.transform(input_data[['Felder']])
    felder_encoded_df = pd.DataFrame(
        felder_encoded,
        columns=onehot_encoder.get_feature_names_out(['Felder'])
    )

    # Escalar 'Examen_admisión_Universidad'
    input_data_scaled = input_data.copy()
    input_data_scaled['Examen_admisión_Universidad_scaled'] = standard_scaler.transform(
        input_data_scaled[['Examen_admisión_Universidad']]
    )

    # Concatenar características
    processed_input = pd.concat([
        input_data_scaled.drop('Felder', axis=1).reset_index(drop=True),
        felder_encoded_df.reset_index(drop=True)
    ], axis=1)

    # Alinear columnas con las del modelo
    if hasattr(model, "feature_names_in_"):
        processed_input = processed_input.reindex(columns=model.feature_names_in_, fill_value=0)
    else:
        expected_columns = [
            'Examen_admisión_Universidad', 'Felder_activo', 'Felder_equilibrio', 'Felder_intuitivo', 
            'Felder_reflexivo', 'Felder_secuencial', 'Felder_sensorial', 'Felder_verbal', 'Felder_visual'
        ]
        processed_input = processed_input.reindex(columns=expected_columns, fill_value=0)

except Exception as e:
    st.error(f"Error en el preprocesamiento: {e}")
    st.stop()

# --- Opcional: mostrar input procesado ---
if st.checkbox("Mostrar datos preprocesados"):
    st.write("Datos procesados para el modelo:")
    st.dataframe(processed_input)

# --- Predicción ---
if st.button('Predecir'):
    try:
        prediction = model.predict(processed_input)
        resultado = "Aprobado" if prediction == 'si' else "No aprobado"
        st.subheader(f"Predicción: {resultado}")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(processed_input)[0]
            prob_aprobado = prob[1]
            st.write(f"Probabilidad de aprobación: **{prob_aprobado:.2%}**")

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")
        st.write("Columnas esperadas por el modelo:", getattr(model, "feature_names_in_", "No definidas"))
        st.write("Columnas del input procesado:", processed_input.columns.tolist())
