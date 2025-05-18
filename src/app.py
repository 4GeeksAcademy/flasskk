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
