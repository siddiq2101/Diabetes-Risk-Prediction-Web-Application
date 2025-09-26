[🚀 **Check Live Demo Here**](https://diabetes-risk-prediction-web-application-bdcgsrp9ikrjlzfkd22jt.streamlit.app/)
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
```
## Notes on Accuracy

In medical prediction projects, achieving very high accuracy (e.g., above 80% or 90%) is often unrealistic and not the primary goal. This is because:

1. **Class Imbalance**: Many medical datasets have fewer positive cases (e.g., patients with a disease) compared to negative cases (healthy patients). High overall accuracy can be misleading if the model mostly predicts the majority class correctly.

2. **Patient Safety Priority**: The main goal in healthcare is to **avoid missing any patients who may have a serious condition**. This often requires lowering the prediction threshold to catch more positive cases, which can reduce overall accuracy but increases recall for the critical cases.

3. **Biological Variability**: Human health is highly variable. Symptoms, test results, and risk factors can differ greatly between individuals, making perfect prediction impossible.

4. **Measurement Errors**: Medical data may contain noise, errors, or missing values due to testing equipment, human recording, or patient reporting, which limits model accuracy.

Hence, in medical ML projects, metrics like **recall, sensitivity, ROC-AUC, and precision-recall trade-offs** are often more important than raw accuracy.

