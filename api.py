from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Approv.io API", description="API for Loan & Credit Approval Predictions")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models and Preprocessors
try:
    credit_model = joblib.load(open('model.pkl', 'rb'))
    credit_preprocessor = joblib.load(open('preprocessor.pkl', 'rb'))
    credit_explainer = joblib.load(open('credit_explainer.pkl', 'rb'))

    loan_model = joblib.load(open('loan_model.pkl', 'rb'))
    loan_preprocessor = joblib.load(open('loan_preprocessor.pkl', 'rb'))
    loan_explainer = joblib.load(open('loan_explainer.pkl', 'rb'))
except Exception as e:
    print(f"Error loading models: {e}")

# --- Helper Functions (copied/adapted from app.py) ---

def scale_debt(user_debt, real_min=0, real_max=10000000, dataset_min=0, dataset_max=28):
    return ((user_debt - real_min) / (real_max - real_min)) * (dataset_max - dataset_min) + dataset_min

def scale_credit_score(user_score, real_min=300, real_max=900, dataset_min=0, dataset_max=67):
    return ((user_score - real_min) / (real_max - real_min)) * (dataset_max - dataset_min) + dataset_min

def scale_income(user_income, real_min=0, real_max=100000000, dataset_min=0, dataset_max=100000):
    return ((user_income - real_min) / (real_max - real_min)) * (dataset_max - dataset_min) + dataset_min

# --- Pydantic Models ---

class CreditApplication(BaseModel):
    Gender: str
    Age: int
    Married: str
    Citizen: str
    Employment: str
    Industry: str
    YearsEmployed: float
    Income: float
    Debt: float
    Bank_Customer: str
    PriorDefault: str
    CreditScore: int
    DriversLicense: str

class LoanApplication(BaseModel):
    loan_amount: float
    loan_term: int
    no_of_dependents: int
    gender: str
    age: int
    education: str
    self_employed: str
    annual_income: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to Approv.io API"}

@app.post("/predict/credit")
def predict_credit(application: CreditApplication):
    try:
        data = pd.DataFrame({
            'Gender': [1 if application.Gender == "Male" else 0],
            'Age': [application.Age],
            'Debt': [scale_debt(application.Debt)],
            'Married': [1 if application.Married == 'Married' else 0],
            'BankCustomer': [1 if application.Bank_Customer == 'Yes' else 0],
            'Industry': [application.Industry],
            'YearsEmployed': [np.log1p(application.YearsEmployed)],
            'PriorDefault': [1 if application.PriorDefault == 'No' else 0],
            'Employed': [1 if application.Employment == 'Yes' else 0],
            'CreditScore': [np.log1p(scale_credit_score(application.CreditScore))],
            'DriversLicense': [1 if application.DriversLicense == 'Yes' else 0],
            'Citizen': [application.Citizen],
            'Income': [np.log1p(scale_income(application.Income))]
        })

        final_data = credit_preprocessor.transform(data)
        prediction = credit_model.predict(final_data)
        confidence = credit_model.predict_proba(final_data)[0].max() * 100
        
        # SHAP explanation (simplified for API)
        feature_names = credit_preprocessor.get_feature_names_out()
        sample = pd.DataFrame(final_data, columns=feature_names)
        shap_values = credit_explainer.shap_values(sample)
        
        feature_importance = []
        for i, col in enumerate(sample.columns):
             feature_importance.append({"feature": col, "value": float(shap_values[0][i])})
        
        feature_importance.sort(key=lambda x: abs(x['value']), reverse=True)

        return {
            "approved": bool(prediction[0]),
            "confidence": float(confidence),
            "top_features": feature_importance[:5]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/loan")
def predict_loan(application: LoanApplication):
    try:
        data = pd.DataFrame({
            'no_of_dependents': [application.no_of_dependents],
            'education': [1 if application.education == 'Yes' else 0],
            'self_employed': [1 if application.self_employed == 'Yes' else 0],
            'annual_income': [application.annual_income],
            'loan_amount': [application.loan_amount],
            'loan_term': [application.loan_term],
            'cibil_score': [application.cibil_score],
            'residential_assets_value': [application.residential_assets_value],
            'commercial_assets_value': [application.commercial_assets_value],
            'luxury_assets_value': [application.luxury_assets_value],
            'bank_asset_value': [application.bank_asset_value]
        })

        final_data = loan_preprocessor.transform(data)
        prediction = loan_model.predict(final_data)
        confidence = loan_model.predict_proba(final_data)[0].max() * 100

        emi = 0
        if prediction[0]:
            rate = 0.08 / 12
            emi = (application.loan_amount * rate * (1 + rate)**application.loan_term) / ((1 + rate)**application.loan_term - 1)

        # SHAP explanation
        feature_names = loan_preprocessor.get_feature_names_out()
        sample = pd.DataFrame(final_data, columns=feature_names)
        shap_values = loan_explainer.shap_values(sample)

        feature_importance = []
        for i, col in enumerate(sample.columns):
             feature_importance.append({"feature": col, "value": float(shap_values[0][i])})
        
        feature_importance.sort(key=lambda x: abs(x['value']), reverse=True)

        return {
            "approved": bool(prediction[0]),
            "confidence": float(confidence),
            "emi": float(emi),
            "top_features": feature_importance[:5]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
