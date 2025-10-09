import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import joblib
import warnings
import xgboost
warnings.filterwarnings('ignore')
import shap 
import matplotlib.pyplot as plt


shap.initjs()

# Page configuration
st.set_page_config(
    page_title="Approv.io - Smart Credit & Loan Approval",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1e3d59;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 10px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 20px 0;
    }
    .danger-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 20px 0;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'credit_data' not in st.session_state:
    st.session_state.credit_data = None
if 'loan_data' not in st.session_state:
    st.session_state.loan_data = None


def predict_credit_approval(data):
    preprocessor = joblib.load(open('preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    model = joblib.load(open('model.pkl', 'rb'))
    prediction = model.predict(final_data)
    confidence = model.predict_proba(final_data)[0].max() * 100
    return prediction[0], confidence
    
def scale_debt(user_debt, real_min=0, real_max=10000000, dataset_min=0, dataset_max=28):
    return ((user_debt - real_min) / (real_max - real_min)) * (dataset_max - dataset_min) + dataset_min  
    
def scale_credit_score(user_score, real_min=300, real_max=900, dataset_min=0, dataset_max=67):
    return ((user_score - real_min) / (real_max - real_min)) * (dataset_max - dataset_min) + dataset_min

def scale_income(user_income, real_min=0, real_max=100000000, dataset_min=0, dataset_max=100000):
    return ((user_income - real_min) / (real_max - real_min)) * (dataset_max - dataset_min) + dataset_min


def predict_loan_approval(data):
    preprocessor = joblib.load(open('loan_preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    model = joblib.load(open('loan_model.pkl', 'rb'))
    prediction = model.predict(final_data)  
    confidence = model.predict_proba(final_data)[0].max() * 100
    return bool(prediction[0]), confidence

def get_loan_explaination(data):
    preprocessor = joblib.load(open('loan_preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()

    sample = pd.DataFrame(final_data,columns=feature_names)
    explainer = joblib.load(open('loan_explainer.pkl','rb'))
    shap_values = explainer(sample)
    return shap_values

def get_loan_top_features(data):
    preprocessor = joblib.load(open('loan_preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()

    sample = pd.DataFrame(final_data,columns=feature_names)
    explainer = joblib.load(open('loan_explainer.pkl','rb'))
    shap_values = explainer.shap_values(sample)

    feature_importance = pd.DataFrame({
        'Feature': sample.columns,
        'Feature Value': shap_values[0]
    }).sort_values(by='Feature Value', key=abs, ascending=False)

    return feature_importance
     
def get_credit_explaination(data):
    preprocessor = joblib.load(open('preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()

    sample = pd.DataFrame(final_data,columns=feature_names)
    explainer = joblib.load(open('credit_explainer.pkl','rb'))
    shap_values = explainer(sample)
    return shap_values

def get_credit_top_features(data):
    preprocessor = joblib.load(open('preprocessor.pkl','rb'))
    final_data = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()

    sample = pd.DataFrame(final_data,columns=feature_names)
    explainer = joblib.load(open('credit_explainer.pkl','rb'))
    shap_values = explainer.shap_values(sample)

    feature_importance = pd.DataFrame({
        'Feature': sample.columns,
        'Feature Value': shap_values[0]
    }).sort_values(by='Feature Value', key=abs, ascending=False)

    return feature_importance

# Home Page
def home_page():
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Approv.io</h1>
        <p>Your One-Stop Platform for Credit & Loan Approvals</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Fast Processing</h3>
            <p>Get instant approval decisions with our AI-powered system</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🔒 Secure & Private</h3>
            <p>Your data is encrypted and protected with industry standards</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h3>📊 Smart Analytics</h3>
            <p>ML-powered decisions for accurate approval predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💳 Credit Card Approval")
        st.write("Check your eligibility for credit card approval with our advanced prediction system.")
        if st.button("Apply for Credit Card", key="credit_btn"):
            st.session_state.page = 'Credit Approval'
            st.rerun()
    
    with col2:
        st.markdown("### 🏠 Loan Approval")
        st.write("Get instant loan approval decisions based on your financial profile.")
        if st.button("Apply for Loan", key="loan_btn"):
            st.session_state.page = 'Loan Approval'
            st.rerun()
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📈 Platform Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Applications Processed", "10,234", "↑ 12%")
    with col2:
        st.metric("Approval Rate", "67%", "↑ 3%")
    with col3:
        st.metric("Average Processing Time", "< 30s", "↓ 5s")
    with col4:
        st.metric("Customer Satisfaction", "4.8/5", "↑ 0.2")

# Credit Approval Page
def credit_approval_page():
    st.markdown("""
    <div class="main-header">
        <h1>💳 Credit Card Approval</h1>
        <p>Fill in your details to check credit card eligibility</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("credit_form"):
        st.markdown("### Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Age = st.number_input("Age", min_value=18, max_value=100, value=30)
            Married = st.selectbox("Marital Status", ["Married", "Single/Divorced/etc"])
            Citizen = st.selectbox('Since when are you a citizen of India',['ByBirth', 'ByOtherMeans', 'Temporary'])
            
            
        
        with col2:
            Employment = st.selectbox("Are you currently employed", ['Yes','No'])
            Industry = st.selectbox("Job sector of current or most recent job", ['Industrials', 'Materials', 'CommunicationServices', 'Transport',
            'InformationTechnology', 'Financials', 'Energy', 'Real Estate',
            'Utilities', 'ConsumerDiscretionary', 'Education',
            'ConsumerStaples', 'Healthcare', 'Research'])
            YearsEmployed = st.number_input("Years Employed", min_value=0, max_value=50, value=2)
            
            
        
        st.markdown("### Financial Information")
        col1, col2 = st.columns(2)
        
        with col1:
            Income = st.number_input("Annual Income",min_value=0,max_value=100000000,value=500000)
            Debt = st.number_input("Outstanding Debt", min_value=0,max_value=10000000,value=0)
            Bank_Customer = st.selectbox("Do you have a bank account?", ['Yes','No'])
        
        with col2:
            PriorDefault = st.selectbox("Any PriorDefault",['Yes','No'])
            CreditScore = st.number_input("Credit Score",min_value=300,max_value=900,value=750)
            DriversLicense = st.selectbox('Do you have a drivers license',['Yes','No'])
            
        
        submitted = st.form_submit_button("Submit Application")
        
        if submitted:
            # Create DataFrame from form data
            data = pd.DataFrame(
                {
                'Gender': [1 if Gender == "Male" else 0],
                'Age': [Age],
                'Debt': [scale_debt(Debt)],
                'Married': [1 if Married == 'Married' else 0],
                'BankCustomer': [1 if Bank_Customer == 'Yes' else 0],
                'Industry' : [Industry],
                'YearsEmployed' : [np.log1p(YearsEmployed)],
                'PriorDefault' : [1 if PriorDefault == 'Yes' else 0],
                'Employed': [1 if Employment == 'Yes' else 0],
                'CreditScore': [np.log1p(scale_credit_score(CreditScore))],
                'DriversLicense': [1 if DriversLicense == 'Yes' else 0],
                'Citizen': [Citizen],
                'Income': [np.log1p(scale_income(Income))]
                }
            )
            
            st.session_state.credit_data = data
            
            # Show loading animation
            with st.spinner('Processing your application...'):
                time.sleep(2)
            
            # Get prediction
            approved, confidence = predict_credit_approval(data)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Application Results")
            
            if approved:
                st.markdown(f"""
                <div class="success-box">
                    <h2>✅ Congratulations! Your Credit Card Application is APPROVED</h2>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()

                st.subheader("Model Explanation for this Prediction")
                values = get_credit_explaination(data)
                plt.figure()
                shap.plots.bar(values)
                st.pyplot(plt)

                st.subheader("Top Factors:")
                feature_importance = get_credit_top_features(data)
                top_features = feature_importance.head(5)
                st.table(top_features)
            else:
                st.markdown(f"""
                <div class="danger-box">
                    <h2>❌ Unfortunately, Your Credit Card Application was Not Approved</h2>
                    <p><strong>Suggestions for Improvement:</strong></p>
                    <ul>
                        <li>Improve your credit history</li>
                        <li>Reduce existing debt obligations</li>
                        <li>Increase your income stability</li>
                        <li>Try again after 6 months</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("Model Explanation for this Prediction")
                values = get_credit_explaination(data)
                plt.figure()
                shap.plots.bar(values)
                st.pyplot(plt)

                st.subheader("Top Factors:")
                feature_importance = get_credit_top_features(data)
                top_features = feature_importance.head(5)
                st.table(top_features)
            
            # Show data summary
            with st.expander("View Application Summary"):
                data = pd.DataFrame(
                {
                'Gender': [Gender],
                'Age': [Age],
                'Debt': [Debt],
                'Married': [Married],
                'BankCustomer': [Bank_Customer],
                'Industry' : [Industry],
                'YearsEmployed' : [YearsEmployed],
                'PriorDefault' : [PriorDefault],
                'Employed': [Employment],
                'CreditScore': [CreditScore],
                'DriversLicense': [DriversLicense],
                'Citizen': [Citizen],
                'Income': [Income]
                }
            )
                st.dataframe(data.T, use_container_width=True)

# Loan Approval Page
def loan_approval_page():
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Loan Approval</h1>
        <p>Check your eligibility for loans</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("loan_form"):
        st.markdown("### Loan Details")
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount Required", min_value=1000, max_value=100000000, value=50000)
            loan_term = st.number_input("Loan Term (months)", min_value=2, max_value=25, value=12)
        
        with col2:
            no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=6,value=1)
        
        st.markdown("### Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            applicant_name = st.text_input("Full Name")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="loan_gender")
            age = st.number_input("Age", min_value=18, max_value=100, value=35, key="loan_age")
            education = st.selectbox("Are you Graduated?",['Yes','No'])
            self_employed = st.selectbox("Are you Self-Employed?",['Yes','No'])
        
        with col2:
            
            annual_income = st.number_input("Annual Income", min_value=0, max_value=100000000, value=600000)
            cibil_score = st.number_input("Cibil Score", min_value=300, max_value=900,value=750)
            residential_assets_value = st.number_input("Estimated value of Residential assets", min_value=0, max_value=300000000,value=1000000)
            commercial_assets_value = st.number_input("Estimated value of Commercial assets", min_value=0, max_value=300000000,value=1000000)
            luxury_assets_value	= st.number_input("Estimated value of Luxury assets", min_value=0, max_value=300000000,value=1000000)
            bank_asset_value = st.number_input("Estimated value of Bank assets", min_value=0, max_value=300000000,value=1000000)
        
        
        submitted = st.form_submit_button("Submit Loan Application")
        
        if submitted:
            # Create DataFrame from form data
            data = pd.DataFrame({
                'no_of_dependents': [no_of_dependents],
                'education': [1 if education == 'Yes' else 0],
                'self_employed': [1 if self_employed == 'Yes' else 0],
                'annual_income': [annual_income],
                'loan_amount': [loan_amount],
                'loan_term': [loan_term],
                'cibil_score': [cibil_score],
                'residential_assets_value': [residential_assets_value],
                'commercial_assets_value': [commercial_assets_value],
                'luxury_assets_value': [luxury_assets_value],
                'bank_asset_value': [bank_asset_value]
            })
            
            st.session_state.loan_data = data
            
            # Show loading animation
            with st.spinner('Analyzing your loan application...'):
                time.sleep(2)
            
            # Get prediction
            approved, confidence = predict_loan_approval(data)
            
            # Calculate EMI
            if approved:
                rate = 0.08 / 12  
                emi = (loan_amount * rate * (1 + rate)**loan_term) / ((1 + rate)**loan_term - 1)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Loan Application Results")
            
            if approved:
                st.markdown(f"""
                <div class="success-box">
                    <h2>✅ Great News! Your Loan Application is APPROVED</h2>
                    <p><strong>Confidence Score:</strong> {confidence:.1f}%</p>
                    <p><strong>Approved Loan Amount:</strong> ₹{loan_amount:,.2f}</p>
                    <p><strong>Estimated Monthly EMI:</strong> ₹{emi:,.2f}</p>
                    <p><strong>Interest Rate:</strong> 8% p.a. </p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                
                # Loan details breakdown
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Loan Amount", f"₹{loan_amount:,.0f}")
                with col2:
                    st.metric("Monthly EMI", f"₹{emi:,.0f}")
                with col3:
                    st.metric("Total Interest", f"₹{(emi * loan_term - loan_amount):,.0f}")
                
                st.subheader("Model Explanation for this Prediction")
                values = get_loan_explaination(data)
                plt.figure()
                shap.plots.bar(values)
                st.pyplot(plt)

                st.subheader("Top Factors:")
                feature_importance = get_loan_top_features(data)
                top_features = feature_importance.head(5)
                st.table(top_features)

            else:
                st.markdown(f"""
                <div class="danger-box">
                    <h2>❌ Your Loan Application Requires Further Review</h2>
                    <p><strong>Confidence Score:</strong> {confidence:.1f}%</p>
                    <p><strong>Reasons for Current Decision:</strong></p>
                    <ul>
                        <li>Debt-to-income ratio needs improvement</li>
                        <li>Credit score below required threshold</li>
                        <li>Insufficient collateral for requested amount</li>
                    </ul>
                    <p><strong>Next Steps:</strong></p>
                    <ul>
                        <li>Consider a co-applicant with stable income</li>
                        <li>Reduce requested loan amount</li>
                        <li>Improve credit score to 750+</li>
                        <li>Clear existing debts to improve eligibility</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("Model Explanation for this Prediction")
                values = get_loan_explaination(data)
                plt.figure()
                shap.plots.bar(values)
                st.pyplot(plt)

                st.subheader("Top Factors:")
                feature_importance = get_loan_top_features(data)
                top_features = feature_importance.head(5)
                st.table(top_features)

                

            # Show data summary
            with st.expander("View Complete Application Data"):
                st.dataframe(data.T, use_container_width=True)

# Dashboard Page
def dashboard_page():
    st.markdown("""
    <div class="main-header">
        <h1>📊 Application Dashboard</h1>
        <p>View your application history and status</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Credit Applications", "Loan Applications"])
    
    with tab1:
        if st.session_state.credit_data is not None:
            st.markdown("### Latest Credit Card Application")
            st.dataframe(st.session_state.credit_data, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Application Status", "Processed")
            with col2:
                st.metric("Processing Time", "< 30 seconds")
            with col3:
                st.metric("Documents Required", "3")
        else:
            st.info("No credit card applications found. Apply for a credit card to see your application details here.")
    
    with tab2:
        if st.session_state.loan_data is not None:
            st.markdown("### Latest Loan Application")
            st.dataframe(st.session_state.loan_data, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Application Status", "Under Review")
            with col2:
                st.metric("Processing Time", "< 30 seconds")
            with col3:
                st.metric("Documents Required", "5")
        else:
            st.info("No loan applications found. Apply for a loan to see your application details here.")

# Main App
def main():
    load_css()
    
    # Sidebar Navigation
    with st.sidebar:
        st.image("logo.png", width='stretch')
        st.markdown("---")
        
        menu_options = ["Home", "Credit Approval", "Loan Approval", "Dashboard", "About"]
        selected_page = st.selectbox("Navigation", menu_options, index=menu_options.index(st.session_state.page))
        st.session_state.page = selected_page
        
        st.markdown("---")
        st.markdown("### 📞 Contact Support")
        st.markdown("Email: support@approv.io")
        st.markdown("Phone: 1-800-900-1")
        
        st.markdown("---")
        st.markdown("### 🔒 Security")
        st.markdown("Your data is encrypted and secure")
        
    # Page Routing
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Credit Approval":
        credit_approval_page()
    elif st.session_state.page == "Loan Approval":
        loan_approval_page()
    elif st.session_state.page == "Dashboard":
        dashboard_page()
    elif st.session_state.page == "About":
        st.markdown("""
        <div class="main-header">
            <h1>About Approv.io</h1>
            <p>Leading the future of digital financial approvals</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Our Mission
        To democratize access to financial services by providing instant, AI-powered approval decisions 
        for credit cards and loans, making the process transparent, fast, and accessible to everyone.
        
        ### Technology Stack
        - **Machine Learning Models**: Advanced algorithms for accurate predictions
        - **Real-time Processing**: Instant decision making
        - **Secure Infrastructure**: Bank-grade security protocols
        - **User-Friendly Interface**: Intuitive design for seamless experience
        
        ### Why Choose Approv.io?
        - ✅ **Instant Decisions**: Get approval status in under 30 seconds
        - ✅ **High Accuracy**: ML models trained on millions of data points
        - ✅ **Transparent Process**: Clear explanation of decisions
        - ✅ **24/7 Availability**: Apply anytime, anywhere
        - ✅ **Data Privacy**: Your information is never shared without consent
        
        ### Partners & Certifications
        - ISO 27001 Certified
        - PCI DSS Compliant
        - GDPR Compliant
        - Partner with 50+ Financial Institutions
        
        ### Version
        v1.0.0 - Released October 2025
        """)

if __name__ == "__main__":
    main()