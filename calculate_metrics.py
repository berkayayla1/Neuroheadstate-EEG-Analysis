import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)

# 1. Load and Prepare the Data
print("Loading data and model...")
try:
    df = pd.read_csv('train.csv')
    model = joblib.load('model.pkl')
except FileNotFoundError:
    print("ERROR: 'train.csv' or 'model.pkl' not found!")
    exit()


X = df.drop('eyeDetection', axis=1)
y = df['eyeDetection']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Make an Prediction and Measure Time (Time Taken)
print("Prediction in process...")
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

time_taken = end_time - start_time

# 3. Calculate Metrics
print("\n--- GERÇEK MODEL PERFORMANSI ---")

# Accuracy (Genel Doğruluk)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy:          {acc:.4f}  (%{acc*100:.2f})")

# Balanced Accuracy (Dengeli Doğruluk)
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {bal_acc:.4f}  (%{bal_acc*100:.2f})")

# Precision (Kesinlik)
prec = precision_score(y_test, y_pred)
print(f"Precision:         {prec:.4f}  (%{prec*100:.2f})")

# Recall (Duyarlılık)
rec = recall_score(y_test, y_pred)
print(f"Recall:            {rec:.4f}  (%{rec*100:.2f})")

# F1 Score (Denge)
f1 = f1_score(y_test, y_pred)
print(f"F1 Score:          {f1:.4f}  (%{f1*100:.2f})")

# ROC AUC (Ayırt Etme Gücü)
try:
    # Some models do not return probability, so we put try-except
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score:     {roc_auc:.4f}")
except:
    # If the model cannot calculate probability (in rare cases) calculate with normal prediction
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC Score:     {roc_auc:.4f}")

# Time Taken (Süre)
print(f"Time Taken:        {time_taken:.4f} seconds (Prediction Time)")

# Confusion Matrix (TP, TN, FP, FN)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("\n--- CONFUSION MATRIX DETAILS ---")
print(f"True Negative (True - Open):  {tn}")
print(f"False Positive (False - Closed): {fp}")
print(f"False Negative (False - Open):   {fn}")
print(f"True Positive (True - Closed):   {tp}")
print("-" * 30)
