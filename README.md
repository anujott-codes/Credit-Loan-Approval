# Approv.io – Credit & Loan Approval System

**Approv.io** is an intelligent, data-driven platform designed to automate and streamline credit and loan approval processes. It leverages advanced machine learning models to provide instant, fair, and explainable decisions.

---

## 🚀 Versions

### **Version 2: Enterprise Web Platform (Latest)**
A modern, full-stack web application featuring a premium React frontend and a robust FastAPI backend. Designed for scalability, user experience, and enterprise-grade performance.

### **Version 1: Streamlit Prototype (Legacy)**
The initial proof-of-concept built entirely in Python using Streamlit.
**🔴 Live Demo (v1):** [https://approvio.streamlit.app/](https://approvio.streamlit.app/)

---

## ❗ Problem Statement

In today’s financial ecosystem, **credit and loan approvals** are often time-consuming, subjective, and prone to human bias. Applicants face long waiting times, and financial institutions struggle with high application volumes, inconsistent evaluation methods, and the risk of default due to poor decision-making.

Key challenges include:
* Manual verification delays
* Inconsistent approval criteria across institutions
* Risk of errors and bias in decision-making
* Limited transparency for applicants

This creates a need for an **automated, data-driven, and transparent solution**.

---

## 💡 Solution

**Approv.io** bridges the gap between applicants and financial institutions by making approvals **faster, fairer, and more reliable**.

*   **Instant Predictions:** Real-time approval status based on ML models.
*   **Explainable AI (SHAP):** Understand *why* a decision was made with feature importance insights.
*   **Data-Driven:** Trained on real-world datasets (XGBoost Classifier).

---

## 🌟 Version 2: Enterprise Web Platform

The latest version of Approv.io transforms the prototype into a production-ready web application.

### 🛠️ Tech Stack (v2)

*   **Frontend:** React, Vite, TailwindCSS, Framer Motion, Lucide React
*   **Backend:** FastAPI, Uvicorn, Python 3.12
*   **ML Engine:** XGBoost, Scikit-learn, SHAP, Pandas, NumPy

### ✨ Key Features (v2)

*   **Premium UI/UX:** A sleek, responsive interface with smooth animations and glassmorphism effects.
*   **REST API Architecture:** Decoupled frontend and backend for scalability.
*   **Interactive Dashboards:** Dynamic result cards with color-coded indicators and confidence scores.
*   **Real-time EMI Calculator:** Instantly estimates monthly payments for approved loans.
*   **Detailed Explanations:** Visual breakdown of key favorable and unfavorable factors.

### 💻 How to Run Version 2 Locally

Follow these steps to set up the project on your local machine.

#### Prerequisites
*   Python 3.12+
*   Node.js 16+

#### 1. Backend Setup (FastAPI)

The backend serves the ML models and API endpoints.

```bash
# Navigate to the project root
cd /path/to/project

# Activate the virtual environment (if not already active)
source venv/bin/activate  # On macOS/Linux
# .\venv\Scripts\activate # On Windows

# Install dependencies (if needed)
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart

# Start the API server
python api.py
```
*The backend will start on `http://localhost:8000`*

#### 2. Frontend Setup (React)

The frontend provides the user interface.

```bash
# Open a new terminal and navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
*The frontend will start on `http://localhost:5173`*

---

## 🧪 Version 1: Streamlit Prototype

The original prototype demonstrating the core ML capabilities.

### 🛠️ Tech Stack (v1)
*   **Language:** Python 3.12
*   **Framework:** Streamlit
*   **ML Libraries:** XGBoost, Scikit-learn, SHAP

### 📊 Features (v1)
*   **Credit Approval:** Predicts credit card eligibility based on financial history.
*   **Loan Approval:** Evaluates loan applications and calculates EMI.
*   **Dashboard:** Visual analytics of historical application trends.
*   **Sidebar Navigation:** Simple access to different modules.

---

## 📈 Machine Learning Pipeline (Shared)

Both versions rely on the same robust ML core:

1.  **Data Preprocessing:**
    *   Encoding categorical variables (`OneHotEncoder`)
    *   Feature scaling (`StandardScaler`)
2.  **Model Training:**
    *   **Credit & Loan Models:** Trained using `XGBClassifier` for high accuracy.
3.  **Explainability:**
    *   **SHAP (SHapley Additive exPlanations):** Used to interpret model predictions and provide transparency.
4.  **Model Serialization:**
    *   Models saved as `.pkl` files using `joblib` for efficient loading.

---

## 👨‍💻 Author

**Anujot Singh**
[@anujott-codes](https://github.com/anujott-codes)

---

## 📎 References

*   **Datasets:**
    *   [Kaggle – Credit Card Approval Data](https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data)
    *   [Kaggle – Loan Approval Prediction Data](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
*   **Tools:** React, FastAPI, Streamlit, Scikit-learn, XGBoost
