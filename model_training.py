# -----------------------
# Import Libraries
# -----------------------
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, accuracy_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
# -----------------------
# 1. Load Data
# -----------------------
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -----------------------
# 2. Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# 3. Apply SMOTE (optional but helps recall)
# -----------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
# -----------------------
# 4. Train XGBoost with class weight to prioritize diabetic detection
# -----------------------
xgb_model = XGBClassifier(
    scale_pos_weight=np.sum(y_train_res == 0) / np.sum(y_train_res == 1),  # weight to balance classes
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=7,
    gamma=5,
    subsample=0.8,
    colsample_bytree=0.6,
    random_state=42,
    eval_metric="auc"
    
)

xgb_model.fit(X_train_res, y_train_res,
              eval_set=[(X_test, y_test)],
              verbose=False)

# -----------------------
# 5. Predict probabilities and adjust threshold for high recall
# -----------------------
y_proba = xgb_model.predict_proba(X_test)[:, 1]

# Shift threshold lower to catch more positives
threshold = 0.35  # lower than 0.5 to increase recall
y_pred = (y_proba >= threshold).astype(int)

# -----------------------
# 6. Evaluation
# -----------------------
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------
# 7. ROC Curve
# -----------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# -----------------------
# 8. Precision-Recall Curve
# -----------------------
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(rec, prec, label="PR Curve")
plt.axvline(x=np.sum(y_pred)/len(y_pred), color='red', linestyle='--', label=f"Threshold={threshold}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()



import joblib

# Save the trained XGBoost model with .joblib extension
joblib.dump(xgb_model, "diabetes_xgb_model.joblib")
print("Model saved as 'diabetes_xgb_model.joblib'")

import joblib

# Load the model
model = joblib.load("diabetes_xgb_model.joblib")

# Example: predict on new data
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(y_pred)
