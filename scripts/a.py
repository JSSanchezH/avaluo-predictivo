import pandas as pd
import joblib
import os

print(os.path.dirname(__file__))
# df_sheet2 = pd.read_excel('./Aprobacion curso 2019.xlsx', sheet_name=1)
# # Load the saved encoder and scaler
# # Assuming 'onehot_encoder.joblib' and 'minmax_scaler.joblib' were saved previously
# loaded_encoder = joblib.load('./onehotencoder.pkl')
# loaded_scaler = joblib.load('./standard_scaler.pkl')

# # Load the best trained RandomForest model
# # Assuming 'random_forest_model.joblib' was saved in cell 8kY3GcblO0kK
# loaded_rf_model = joblib.load('./rf_model.pkl')

# # Preprocess df_sheet2: One-hot encode 'Felder'
# # Select the 'Felder' column for encoding
# felder_column_sheet2 = df_sheet2[['Felder']]

# # Transform the 'Felder' column using the loaded encoder
# df_sheet2_encoded = loaded_encoder.transform(felder_column_sheet2)

# # Create a DataFrame from the encoded features
# encoded_feature_names = loaded_encoder.get_feature_names_out(['Felder'])
# df_sheet2_encoded_df = pd.DataFrame(df_sheet2_encoded, columns=encoded_feature_names, index=df_sheet2.index)

# # Concatenate the encoded features with the numerical feature
# # Need to drop the original 'Felder' column first
# df_sheet2_processed = df_sheet2.drop('Felder', axis=1)
# df_sheet2_processed = pd.concat([df_sheet2_processed, df_sheet2_encoded_df], axis=1)

# # Drop the original 'ID' and 'Año - Semestre' columns as they were dropped from the training data
# df_sheet2_processed = df_sheet2_processed.drop(['ID', 'Año - Semestre'], axis=1)

# # Preprocess df_sheet2: Scale 'Examen_admisión'
# # Assuming 'Examen_admisión' in df_sheet2 corresponds to 'Examen_admisión_Universidad' in the training data
# # Fix: Rename the scaled column to match the training data column name
# df_sheet2_processed['Examen_admisión_Universidad_scaled'] = loaded_scaler.transform(df_sheet2_processed[['Examen_admisión']])


# # Drop the original unscaled column
# df_sheet2_processed = df_sheet2_processed.drop('Examen_admisión', axis=1)

# # Ensure the columns in the processed df_sheet2 match the training data columns order
# # Based on X_resampled from cell qcQ6tsY4Hicd and the processed new_data_processed from cell -uWHKcWO_EiA,
# # the columns order expected by the model is:
# # 'Felder_activo', 'Felder_equilibrio', 'Felder_intuitivo',
# # 'Felder_reflexivo', 'Felder_secuencial', 'Felder_sensorial',
# # 'Felder_verbal', 'Felder_visual', 'Examen_admisión_Universidad_scaled'

# # Create a list of columns in the order expected by the model
# model_columns = ['Felder_activo', 'Felder_equilibrio', 'Felder_intuitivo',
#                  'Felder_reflexivo', 'Felder_secuencial', 'Felder_sensorial',
#                  'Felder_verbal', 'Felder_visual', 'Examen_admisión_Universidad_scaled']

# # Reindex the processed df_sheet2 DataFrame to match the model's expected column order
# df_sheet2_processed = df_sheet2_processed.reindex(columns=model_columns, fill_value=0)


# # Make predictions using the loaded RandomForest model
# predictions_sheet2 = loaded_rf_model.predict(df_sheet2_processed)

# # Add the predictions to the original df_sheet2 DataFrame for easy viewing
# df_sheet2['Predicted_Aprobo_Calculo'] = predictions_sheet2
# # Display the updated df_sheet2 with predictions
# print(df_sheet2_processed)