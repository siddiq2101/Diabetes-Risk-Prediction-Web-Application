# Diabetes-Risk-Prediction-Web-Application
A web application that predicts the risk of diabetes using machine learning. The app leverages an **XGBoost classifier** trained on the **Pima Indians Diabetes Dataset** and provides real-time risk assessment based on user-provided health parameters.
---

## Files in the repository

| File | Description |
|------|-------------|
| `app.py` | Streamlit web application for predicting diabetes risk. |
| `model_training.py` | Python script to train the XGBoost model on the diabetes dataset. |
| `diabetes.csv` | **Pima Indians Diabetes Dataset** used for training the model. |
| `diabetes_xgb_model.joblib` | Trained XGBoost model saved using Joblib. |
| `requirements.txt` | Lists all Python dependencies needed to run the app and training scripts. |
---

## Dataset

The project uses the **Pima Indians Diabetes Dataset** from the UCI Machine Learning Repository.  

**Features in the dataset:**

- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age (years)
- `Outcome`: Class variable (0: Non-diabetic, 1: Diabetic)

---

## Features

- Predicts **high or low risk** of diabetes.
- Provides **probability** of diabetes.
- Uses **SMOTE** to handle imbalanced dataset and improve detection of diabetic cases.
- Prioritizes **recall for diabetic cases** to reduce false negatives.

---

## Requirements

Make sure you have Python 3.8+ installed. Install the required packages:

```bash
pip install -r requirements.txt
