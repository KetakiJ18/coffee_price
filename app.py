import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import os
import warnings
from pycaret.regression import load_model, predict_model

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the trained PyCaret model
model = load_model("coffee_sales_rf_model")

# Correct column names EXACTLY as trained
feature_order = [
    "hour_of_day",
    "coffee_name",
    "Time_of_Day",
    "Weekday",
    "Month_name"
]

def preprocess_input(data_dict):
    """Convert form/JSON input into a pandas DataFrame with correct columns"""
    df = pd.DataFrame([{
        "hour_of_day": int(data_dict["hour_of_day"]),
        "coffee_name": data_dict["coffee_name"],
        "Time_of_Day": data_dict["Time_of_Day"],
        "Weekday": data_dict["Weekday"],
        "Month_name": data_dict["Month_name"]
    }])
    return df

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    df = preprocess_input(data)
    result = predict_model(model, data=df)
    output = result["prediction_label"][0]
    return jsonify(float(output))

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form inputs
    data = {col: request.form[col] for col in feature_order}

    # Preprocess → DataFrame
    df = preprocess_input(data)

    # PyCaret uses predict_model or model.predict
    result = predict_model(model, df)
    output = result["prediction_label"][0]

    prediction_text = "Predicted Sales: ₹" + str(round(float(output), 2))
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))