# ☕ Coffee Price Prediction 

This project predicts **coffee prices** based on features like:
* `hour_of_day`
* `time_of_day`
* `coffee_name`
* `day_of_week`
* `month`

A **Random Forest Regression model** is used for prediction, with an accuracy of **97%**.
The pipeline is fully automated using **PyCaret**, **GitHub Actions**, **Docker**, and **Render**.

---

Check it out! [Coffee Price Prediction](https://coffee-price-prediction-60ns.onrender.com)

## Features
### Machine Learning (PyCaret)
* Automated model training & comparison using `pycaret.regression`.
* Best model automatically selected and saved.
* Model reloads inside Flask for real-time predictions.

### CI/CD with GitHub Actions
* Whenever `coffee_sales.csv` changes, the pipeline:
  1. Retrains the model
  2. Saves the updated best model
  3. Builds and pushes the Docker image
  4. Deploys to Render via Web Service

### Flask App
* API endpoint for predictions
* Web UI built with simple HTML/CSS for user interaction

### Docker + Render Deployment
* Dockerfile used to containerize the entire app
* Render automatically builds & deploys using the Dockerfile
* Fully cloud-hosted and reproducible

---

## Project Structure
```
.
├── app.py                      # Flask backend
├── coffee_sales_rf_model.pkl   # Saved ML model (auto-updated)
├── coffee_sales.csv            # Training data (updates trigger pipeline)
├── templates/
│   └── index.html              # Frontend UI
├── Dockerfile                  # Used by Render deployment
├── requirements.txt            # Project dependencies
└── .github/
    └── workflows/
        └── train.yml           # GitHub Actions pipeline
```

---

## How It Works
### Model Training
PyCaret handles:
* Data preprocessing
* Model comparison
* Selection of the best-performing regressor
* Export of the final model

### Automated CI/CD
GitHub Actions workflow:
* Detects change in `coffee_sales.csv`
* Retrains model using PyCaret
* Updates `coffee_sales_rf_model.pkl`
* Builds Docker image
* Deploys to Render Web Service

### Flask Endpoints
```
/        -> Web UI  
/predict -> JSON API for model predictions  
```

---

## Docker Setup (Local)
### Build
```bash
docker build -t coffee-price-app .
```
### Run
```bash
docker run -p 5000:5000 coffee-price-app
```

---

## Environment Requirements
* Python 3.9-3.11
* PyCaret (Regression)
* Flask
* scikit-learn
* Pandas / NumPy
* Render Web Service
* GitHub Actions

---

## Web UI Preview

<img width="1280" height="894" alt="image" src="https://github.com/user-attachments/assets/53afd9ca-cc40-4668-92b3-e0d92b8db887" />
