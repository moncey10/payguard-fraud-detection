# 🛡️ PayGuard — Payment Fraud Detection System

> A full-stack ML-powered fraud detection system inspired by real-world payment gateway architectures like Razorpay, PayU, and Cashfree.



---

## 📌 Overview

PayGuard is an end-to-end credit card fraud detection system that analyzes transactions in real-time using a machine learning model trained on 284,807 transactions. The system consists of:

- **ML Model** — Logistic Regression with custom F1-optimized threshold
- **REST API** — FastAPI backend serving predictions in under 50ms
- **User Frontend** — Payment gateway-style transaction testing UI
- **Admin Dashboard** — Full admin panel with live monitoring, batch CSV analysis, and manual transaction testing

---

## 🚀 Features

### ML Model
- Trained on the Kaggle Credit Card Fraud Detection dataset (284,807 transactions)
- Handles extreme class imbalance (0.17% fraud ratio) using `class_weight='balanced'`
- Custom threshold optimization via F1 score instead of default 0.5
- StandardScaler + Logistic Regression sklearn Pipeline
- **ROC-AUC: 97.2%** | **Avg Precision: 73.8%**

### FastAPI Backend
- `POST /predict` — Real-time fraud prediction with probability score
- `GET /fraud_sample` — Sample fraud transaction for testing
- `GET /` — Health check endpoint
- CORS enabled for frontend integration
- Supports both Pydantic v1 and v2

### User Frontend (`index.html`)
- Razorpay-inspired payment gateway UI
- Single transaction test with probability score
- Batch CSV upload — analyze multiple transactions at once
- Real-time BLOCKED / APPROVED result with risk level

### Admin Dashboard (`admin.html`)
- 🔐 Login protected (admin/admin)
- 📊 Overview — KPI cards, recent transactions, fraud rate gauge
- 💳 Transactions — Complete audit trail
- ⚡ Live Test
  - **Manual Entry** — Enter card details, model analyzes
  - **CSV Upload** — Batch analyze all rows at once
- 🤖 Model Info — Configuration and performance metrics
- 📋 Audit Logs — System event log

---

## 🗂️ Project Structure

```
payguard-fraud-detection/
│
├── app.py                  # FastAPI application
├── preprocessing.py        # Model training & threshold optimization
├──Pipeline/
    ├── build_pipeline.py       # Sklearn pipeline definition
├── requirements.txt        # Python dependencies
│
├── index.html              # User-facing frontend
├── admin.html              # Admin dashboard
│
├── artifacts/               # Generated after training (git ignored)
│   ├── model.pkl
│   └── threshold.pkl
│
└── data/                   # Dataset (git ignored)
    └── creditcard.csv
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/moncey10/payguard-fraud-detection.git
cd payguard-fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

```
data/
└── creditcard.csv
```

### 4. Train the model

```bash
python preprocessing.py
```

This will generate `artifacts/model.pkl` and `artifacts/threshold.pkl`.

Expected output:
```
Best threshold: 0.14 with F1: 0.xxxx
ROC-AUC: 0.97xx
Average Precision: 0.73xx
Model saved!
```

### 5. Start the API server

```bash
uvicorn app:app --reload
```

API will be available at `http://127.0.0.1:8000`

### 6. Open the frontend

Open `index.html`  directly in your browser.

Admin login credentials: `admin` / `admin`

---

## 🔌 API Reference

### `POST /predict`

Predict whether a transaction is fraudulent.

**Request Body:**
```json
{
  "V1": -1.3598,
  "V2": -0.0728,
  "V3": 2.5363,
  ...
  "V28": -0.0211
}
```

**Response:**
```json
{
  "prediction": 0,
  "fraud": false,
  "probability": 0.1365,
  "message": "Legitimate transaction"
}
```

### `GET /fraud_sample`

Returns a real fraud transaction sample from the dataset for testing.

### `GET /`

Health check — returns API status.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 97.2% |
| Average Precision | 73.8% |
| Threshold | 0.14 (F1-optimized) |
| Training Set | 227,846 transactions |
| Test Set | 56,961 transactions |
| Fraud Ratio | 0.17% (highly imbalanced) |

---

## 🧠 Architecture

```
User / Admin
     │
     ▼
Frontend (HTML/CSS/JS)
     │  POST /predict
     ▼
FastAPI Backend (app.py)
     │
     ▼
ML Pipeline
  ├── StandardScaler
  └── Logistic Regression
          │
          ▼
    Fraud Probability Score
          │
    Custom Threshold (0.14)
          │
    BLOCKED / APPROVED
```

---

## 🔍 How It Works

1. Transaction data is received by the FastAPI `/predict` endpoint
2. Data is passed through a StandardScaler + Logistic Regression pipeline
3. Model outputs a fraud probability score (0 to 1)
4. Score is compared against the optimized threshold (0.14)
5. Transaction is **BLOCKED** if score ≥ threshold, else **APPROVED**

> **Note:** The 28 features (V1–V28) are PCA-transformed signals derived from real transaction data. In a production system, these would be computed internally by the payment gateway from card details, device fingerprint, velocity signals, and geolocation data.

---

## 🧪 Testing

### Single Transaction Test
Open `index.html` → Select a scenario → Click **Run Fraud Detection**

### Admin Manual Test
Open `admin.html` → Login → Live Test → Manual Entry → Fill card details → Click **Analyze Transaction**

### Batch CSV Test
Open `admin.html` → Login → Live Test → CSV Upload → Upload `test_batch.csv` → Click **Analyze All Transactions**

A sample test CSV with mixed legitimate and fraudulent transactions can be generated:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/creditcard.csv')
legit = df[df['Class'] == 0].sample(14, random_state=42)
fraud = df[df['Class'] == 1].sample(6, random_state=42)
test_df = pd.concat([legit, fraud]).sample(frac=1, random_state=42)
test_df = test_df.drop(['Class', 'Time', 'Amount'], axis=1)
test_df.to_csv('test_batch.csv', index=False)
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
fastapi
uvicorn
pydantic
```

---

## 🙋 Author

**Moncey** — Final Year IT Student, Gujarat Technological University

---

## 📄 License

This project is licensed under the MIT License.

---

> ⚡ Built as a portfolio project demonstrating end-to-end ML system design, REST API development, and payment gateway-inspired UI — not affiliated with Razorpay Pvt. Ltd.
