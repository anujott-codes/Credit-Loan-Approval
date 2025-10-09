# Approv.io – Credit & Loan Approval System

**Live App:** [https://approvio.streamlit.app/](https://approvio.streamlit.app/)

---

## ❗ Problem Statement

In today’s financial ecosystem, **credit and loan approvals** are often time-consuming, subjective, and prone to human bias. Applicants face long waiting times, and financial institutions struggle with high application volumes, inconsistent evaluation methods, and the risk of default due to poor decision-making.

Key challenges include:

* Manual verification delays
* Inconsistent approval criteria across institutions
* Risk of errors and bias in decision-making
* Limited transparency for applicants

This creates a need for an **automated, data-driven, and transparent solution** to streamline financial approval systems.

---

## 💡 Solution – Introducing Approv.io

**Approv.io** is an **end-to-end machine learning solution** that automates credit and loan approvals. By leveraging the **XGBoost Classifier** for both credit and loan approvals and integrating **SHAP-based explainability**, Approv.io ensures that every decision is **data-driven, fast, and transparent**.

**Key Highlights:**

* **Instant Predictions** – Applicants get real-time approval results.
* **Data-Driven Decisions** – Trained on real-world datasets to ensure accuracy and fairness.
* **Explainable AI (SHAP)** – Provides interpretability into how features affect predictions.
* **User-Friendly Interface** – Built with Streamlit for simplicity and accessibility.

In short, Approv.io bridges the gap between applicants and financial institutions by making approvals **faster, fairer, and more reliable**.

---

## 📌 Overview

**Approv.io** is a complete end-to-end ML project designed to automate **Credit Approval** and **Loan Approval** processes using **XGBoost models** for both systems.

The platform provides an intuitive web interface where users can input relevant financial details and instantly receive predictions along with **model explainability insights (via SHAP plots)**.

---

## 🎯 Objectives

* Automate decision-making for credit and loan approvals.
* Provide fast, reliable ML predictions.
* Deliver model transparency with SHAP-based interpretability.
* Demonstrate a **production-ready ML pipeline** deployed on the cloud.
* Offer a seamless user experience with **Streamlit UI**.

---

## ⚙️ Tech Stack

### 🔹 Programming Languages

* **Python 3.12** – Core development language

### 🔹 Libraries & Frameworks

* **Machine Learning:**

  * `xgboost` – Training gradient boosting models for both credit and loan approvals
  * `scikit-learn` – Preprocessing, metrics, and pipeline support
  * `shap` – Explainable AI (SHAP value visualizations)
  * `joblib` – Model serialization

* **Data Analysis & Processing:**

  * `pandas` – Data manipulation
  * `numpy` – Numerical computations

* **Visualization:**

  * `matplotlib`
  * `seaborn`

* **Web Application:**

  * `streamlit` – Interactive UI and deployment

### 🔹 Deployment

* **Platform:** Streamlit Community Cloud
* **App URL:** [https://approvio.streamlit.app/](https://approvio.streamlit.app/)
* **Environment:**

  * `requirements.txt` includes all dependencies
  * Models pre-trained in **Google Colab** and saved as `.pkl` files

---

## 📊 Features

### 1. **Credit Approval System (XGBoost Model)**

* **Inputs:** Gender, Age, Debt, Married, BankCustomer, Industry, YearsEmployed, PriorDefault, Employed, CreditScore, Income, Citizen, DriversLicense
* **Model:** XGBoost Classifier
* **Output:** Approved / Rejected decision
* **Explainability:** SHAP summary and force plots for transparency

### 2. **Loan Approval System (XGBoost Model)**

* **Inputs:** loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, loan_status
* **Model:** XGBoost Classifier
* **Output:** Approved / Rejected decision + EMI calculation (if approved)
* **Explainability:** SHAP plots to visualize feature importance for each prediction

### 3. **Dashboard**

* Displays previous application results (simulated dataset + session data).
* Visual analytics for overall trends.

### 4. **User Interface**

* Sidebar navigation (Home, Credit Approval, Loan Approval, Dashboard, About).
* Clean design with structured forms for user input.

---

## 📈 Machine Learning Pipeline

1. **Data Preprocessing**

   * Encoding categorical variables (`OneHotEncoder`)
   * Feature scaling (`StandardScaler`)

2. **Model Training**

   * **Credit Approval:** Trained with `XGBClassifier`
   * **Loan Approval:** Trained with `XGBClassifier`

3. **Explainable AI Integration**

   * SHAP explainer integrated into both models
   * Force plots and summary plots to interpret model predictions

4. **Evaluation Metrics**

   * Accuracy
   * Precision, Recall, F1-Score
   * Confusion Matrix

5. **Model Storage**

   * Trained models saved as `.pkl` files with `joblib`

---

## 🚀 Deployment Process

1. Model training in Google Colab → export `.pkl` models
2. App development in **Streamlit** with modular pages
3. Integration of SHAP for explainability visualizations
4. Dependency management with `requirements.txt`
5. Deployment to **Streamlit Community Cloud**

---

## 🔐 Security & Limitations

* Static ML models (no live retraining yet)
* User data stored only in session (not persistent)
* SHAP visualizations may be computationally intensive for large datasets

---

## 📅 Future Improvements

* Integrate real-time database (PostgreSQL / Firebase)
* Implement authentication for personalized dashboards
* Expand into other financial services (e.g., insurance approvals)
* Add API endpoints for third-party integration

---

## 👨‍💻 Author

**Anujot Singh**
[@anujott-codes](https://github.com/anujott-codes)

---

## 📎 References

* **Datasets:**

  * Credit dataset: [Kaggle – Credit Card Approval Data](https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data)
  * Loan dataset: [Kaggle – Loan Approval Prediction Data](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
* **Libraries:** scikit-learn, xgboost, shap, pandas, numpy
* **Deployment:** [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🌐 Live Demo

👉 Try the app here: [https://approvio.streamlit.app/](https://approvio.streamlit.app/)
