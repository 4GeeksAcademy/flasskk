#STREAMLIT
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo
model = joblib.load("/workspaces/flasskk/models/modelo_gold.pkl")

# Interfaz
st.set_page_config(page_title="Predicción del Precio del Oro", layout="centered")
st.title("Predicción del Precio del Oro")
st.markdown("Introduce los valores de mercado para obtener una predicción.")

# Inputs en columnas
col1, col2 = st.columns(2)
with col1:
    open_val = st.number_input("Open", min_value=0.0, format="%.2f")
    high_val = st.number_input("High", min_value=0.0, format="%.2f")
    low_val = st.number_input("Low", min_value=0.0, format="%.2f")
with col2:
    close_val = st.number_input("Close", min_value=0.0, format="%.2f")
    volume_val = st.number_input("Volume", min_value=0.0, format="%.0f")

# Predicción
if st.button("Predecir"):
    features = [[open_val, high_val, low_val, close_val, volume_val]]
    prediction = model.predict(features)[0]
    st.success(f"Predicción del precio futuro: **{prediction:.2f}**")

# Gráfico
if st.checkbox("Ejemplo de evolución histórica"):
    # Simulación de datos
    df = pd.DataFrame({
        'Año': np.arange(2015, 2025),
        'Precio': np.linspace(1100, 2100, 10) + np.random.normal(0, 50, 10)
    })
    st.line_chart(df.set_index('Año'))
