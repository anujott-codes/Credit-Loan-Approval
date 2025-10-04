# Approv.io – Credit & Loan Approval System

**Live App:** [https://approvio.streamlit.app/](https://approvio.streamlit.app/)

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

* Inputs:

  * Gender, Age, Debt, Married, BankCustomer, Industry, YearsEmployed, PriorDefault, Employed, CreditScore, Income, Citizen, DriversLicense
* Model: **Support Vector Classifier (SVC)**
* Output:

  * **Approved / Rejected** decision

### 2. **Loan Approval System (XGBoost Model)**

* Inputs:

  * loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, loan_status
* Model: **XGBoost Classifier**
* Output:

  * **Approved / Rejected** decision
  * EMI calculation (if approved)

### 3. **Dashboard**

* Displays previous applications & their results (simulated dataset + session data).

### 4. **User Interface**

* **Navigation Sidebar** for page selection (Home, Credit Approval, Loan Approval, Dashboard, About).
* Responsive and minimal design using Streamlit components.
* Clear instructions & structured input forms.

---

## 📈 Machine Learning Pipeline

### 1. **Data Preprocessing**

* Encoding categorical variables (`OneHotEncoder`)
* Feature scaling (`StandardScaler`)

### 2. **Model Training**

* **Credit Approval:**

  * Trained with `SVC`
* **Loan Approval:**

  * Trained with `XGBClassifier`

### 3. **Evaluation Metrics**

* Accuracy
* Precision, Recall, F1-Score
* Confusion Matrix

### 4. **Model Storage**

* Models saved as `.pkl` files using `joblib` for production inference.

---

## 🚀 Deployment Process

1. **Model Training** – Performed in Google Colab, exported trained `.pkl` models.
2. **App Development** – Built using **Streamlit** with modular pages.
3. **Dependency Management** – All required libraries listed in `requirements.txt`.
4. **Deployment** – Pushed to GitHub, connected to **Streamlit Community Cloud**, deployed live.

---

## 🔐 Security & Limitations

* Current version uses static ML models without live database integration.
* User data is processed within the session (not stored permanently).
* Model interpretability is limited; future versions may integrate **SHAP** for explainable AI.

---

## 📅 Future Improvements

* Add **Explainable AI (XAI)** with SHAP values for transparency.
* Integrate **real-time database (PostgreSQL / Firebase)** to store applications.
* Implement **authentication system** for personalized dashboards.
* Extend to other financial products (e.g., insurance approvals).

---

## 👨‍💻 Author

**Anujot Singh**
@anujott-codes

---

## 📎 References

* Dataset: UCI Machine Learning Repository, Kaggle
* Dataset Links
  - Credit dataset : https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data
  - Loan dataset : https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
* Libraries: scikit-learn, xgboost, pandas, numpy
* Deployment: [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🌐 Live Demo

👉 Try the app here: [https://approvio.streamlit.app/](https://approvio.streamlit.app/)

