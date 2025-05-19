from dotenv import load_dotenv
import os , joblib
from flask import Flask, request, render_template
from pickle import load

load_dotenv()

API_KEY = os.getenv("API_KEY")
print(f"API_KEY: {API_KEY}")
app = Flask(__name__)
# Cargar el modelo
model_path = "../models/modelo_gold.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print("Error: El archivo del modelo no se encuentra.")
    model = None

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":

        # Obtain values from form
        val1 = float(request.form.get("val1", 0))
        val2 = float(request.form.get("val2", 0))
        val3 = float(request.form.get("val3", 0))
        val4 = float(request.form.get("val4", 0))
        val5 = float(request.form.get("val5", 0))

        data = [[val1, val2, val3, val4, val5]]
    
        prediction = str(model.predict(data)[0])
        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=None)

#STREAMLIT
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo
model = joblib.load("modelo_gold.pkl")

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
if st.button("🔍 Predecir"):
    features = [[open_val, high_val, low_val, close_val, volume_val]]
    prediction = model.predict(features)[0]
    st.success(f"Predicción del precio futuro: **{prediction:.2f}**")

# Extra: Gráfico (opcional)
if st.checkbox("Ejemplo de evolución histórica"):
    # Simulación de datos
    df = pd.DataFrame({
        'Año': np.arange(2015, 2025),
        'Precio': np.linspace(1100, 2100, 10) + np.random.normal(0, 50, 10)
    })
    st.line_chart(df.set_index('Año'))
