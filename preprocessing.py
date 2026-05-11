import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/creditcard.csv')

x = df.drop(['Class', 'Time', 'Amount'], axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])
pipeline.fit(x_train, y_train)

# find best threshold
proba = pipeline.predict_proba(x_test)[:, 1]
thresholds = np.arange(0.05, 0.5, 0.01)
best_thresh = 0.5
best_f1 = 0

from sklearn.metrics import f1_score
for t in thresholds:
    preds = (proba >= t).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best threshold: {best_thresh:.2f} with F1: {best_f1:.4f}")

os.makedirs('artifacts', exist_ok=True)

with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# save threshold too
with open('artifacts/threshold.pkl', 'wb') as f:
    pickle.dump(float(best_thresh), f)

y_pred = (proba >= best_thresh).astype(int)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, proba)
avg_precision = average_precision_score(y_test, proba)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
print(f"Confusion Matrix:\n{cm}")
print("\nModel saved!")