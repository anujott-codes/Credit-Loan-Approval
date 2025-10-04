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

**Approv.io** is an **end-to-end machine learning solution** that automates credit and loan approvals. By leveraging **Support Vector Classifier (SVC)** for credit approval and **XGBoost** for loan approval, Approv.io provides:

* **Instant Predictions** – Applicants get real-time results.
* **Data-Driven Decisions** – Models trained on real-world datasets ensure consistency.
* **Transparency** – Clear output with the option to extend into explainable AI.
* **User-Friendly Interface** – Built with Streamlit, the platform is intuitive and accessible.

In short, Approv.io bridges the gap between applicants and financial institutions by making approvals **faster, fairer, and more reliable**.

---

## 📌 Overview

**Approv.io** is a complete end-to-end machine learning project designed to automate **Credit Approval** and **Loan Approval** decisions. The app leverages two trained ML models:

* **Credit Approval Model** – Support Vector Classifier (**SVC**)
* **Loan Approval Model** – Extreme Gradient Boosting (**XGBoost**)

The platform provides an intuitive web interface where users can input relevant financial details and instantly receive predictions regarding their credit/loan application outcomes.

---

## 🎯 Objectives

* Automate decision-making for credit and loan approvals.
* Provide fast, reliable ML predictions.
* Demonstrate a **production-ready ML pipeline** deployed on the cloud.
* Deliver an interactive UI using **Streamlit** for a seamless user experience.

---

## ⚙️ Tech Stack

### 🔹 Programming Languages

* **Python 3.12** – Core development language

### 🔹 Libraries & Frameworks

* **Machine Learning:**

  * `scikit-learn` – Training SVC model for credit approval
  * `xgboost` – Training gradient boosting model for loan approval
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

### 1. **Credit Approval System (SVC Model)**

* **Inputs:** Gender, Age, Debt, Married, BankCustomer, Industry, YearsEmployed, PriorDefault, Employed, CreditScore, Income, Citizen, DriversLicense
* **Model:** Support Vector Classifier (SVC)
* **Output:** Approved / Rejected decision

### 2. **Loan Approval System (XGBoost Model)**

* **Inputs:** loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, loan_status
* **Model:** XGBoost Classifier
* **Output:** Approved / Rejected decision + EMI calculation (if approved)

### 3. **Dashboard**

* Displays previous applications & their results (simulated dataset + session data).

### 4. **User Interface**

* Sidebar navigation (Home, Credit Approval, Loan Approval, Dashboard, About).
* Clean design with structured forms for user input.

---

## 📈 Machine Learning Pipeline

1. **Data Preprocessing**

   * Encoding categorical variables (`OneHotEncoder`)
   * Feature scaling (`StandardScaler`)

2. **Model Training**

   * **Credit Approval:** Trained with `SVC`
   * **Loan Approval:** Trained with `XGBClassifier`

3. **Evaluation Metrics**

   * Accuracy
   * Precision, Recall, F1-Score
   * Confusion Matrix

4. **Model Storage**

   * Trained models saved as `.pkl` files with `joblib`

---

## 🚀 Deployment Process

1. Model training in Google Colab → export `.pkl` models
2. App development in **Streamlit** with modular pages
3. Dependency management with `requirements.txt`
4. Deployment to **Streamlit Community Cloud**

---

## 🔐 Security & Limitations

* Static ML models (no live retraining yet)
* User data stored only in session (not persistent)
* Limited interpretability (future integration of SHAP planned)

---

## 📅 Future Improvements

* Add **Explainable AI (SHAP values)** for transparency
* Integrate real-time database (PostgreSQL / Firebase)
* Implement authentication for personalized dashboards
* Expand to other financial services (e.g., insurance approvals)

---

## 👨‍💻 Author

**Anujot Singh**
[@anujott-codes](https://github.com/anujott-codes)

---

## 📎 References

* **Datasets:**

  * Credit dataset: [Kaggle – Credit Card Approval Data](https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data)
  * Loan dataset: [Kaggle – Loan Approval Prediction Data](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
* **Libraries:** scikit-learn, xgboost, pandas, numpy
* **Deployment:** [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🌐 Live Demo

👉 Try the app here: [https://approvio.streamlit.app/](https://approvio.streamlit.app/)
