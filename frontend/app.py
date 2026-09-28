"""
Streamlit Frontend Interface for CECL Credit Risk Assessment.
Provides user input fields for loan parameters and routes HTTP requests to the decoupled FastAPI backend
to retrieve predictive ML inferences and AI-generated governance documentation.
"""
import os
import requests
import streamlit as st

# Define Backend Service Endpoint via Environment Variables
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Initialize UI Configuration
st.set_page_config(layout="wide", page_title="CECL & RAG Dashboard")
st.title("🏦 CECL Credit Risk & AI Governance Dashboard")

st.markdown("""
*This application implements a decoupled microservices architecture. The Streamlit frontend routes HTTP requests to a FastAPI backend hosting XGBoost/Random Forest models.*
**Note:** The backend service is hosted on a scaled-to-zero free cloud tier. Initial request latency may be elevated (~50s) during server cold starts.
""")
st.divider()

# Render layout in a dual-column format
col1, col2 = st.columns(2)

with col1:
    st.header("1. Loan Parameter Configuration")
    
    # Input fields for borrower financial profile
    loan_amnt = st.number_input("Loan Amount ($)", value=15000.0)
    int_rate = st.number_input("Interest Rate (%)", value=12.5)
    installment = st.number_input("Monthly Installment", value=450.0)
    annual_inc = st.number_input("Annual Income ($)", value=65000.0)
    dti = st.number_input("Debt-to-Income (DTI)", value=18.5)
    delinq_2yrs = st.number_input("Delinquencies (Last 2 yrs)", value=0.0)
    open_acc = st.number_input("Open Accounts", value=10.0)
    pub_rec = st.number_input("Public Records (Bankruptcies)", value=0.0)
    revol_util = st.number_input("Revolving Utilization (%)", value=55.0)
    total_acc = st.number_input("Total Accounts", value=22.0)

    if st.button("Predict Expected Credit Loss (CECL)", use_container_width=True):
        payload = {
            "loan_amnt": loan_amnt, "int_rate": int_rate, "installment": installment,
            "annual_inc": annual_inc, "dti": dti, "delinq_2yrs": delinq_2yrs,
            "open_acc": open_acc, "pub_rec": pub_rec, "revol_util": revol_util, "total_acc": total_acc
        }
        
        with st.spinner("Executing Inference via FastAPI Backend..."):
            try:
                res = requests.post(f"{API_URL}/predict_cecl", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.success("Inference Complete.")
                    st.metric("Probability of Default (PD)", f"{data['PD']*100:.2f}%")
                    st.metric("Loss Given Default (LGD)", f"{data['LGD']*100:.2f}%")
                    st.metric("Total Expected Loss (CECL)", f"${data['Expected_Credit_Loss']:,.2f}")
                else:
                    st.error(f"Service Error: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Backend Connection Error: {e}")

with col2:
    st.header("2. AI Model Risk Governance (MRM)")
    
    # Dynamic selection of foundational LLM architecture
    selected_model = st.selectbox(
        "Select Target LLM Architecture", 
        ["Groq (LLaMA 3.1 8B)", "Groq (LLaMA 3.3 70B)", "Google (Gemini)"],
        help="Routes query to specific language model backend for generation. Groq LLaMA models are highly recommended for lowest latency."
    )
    
    st.markdown(f"Initialize pipeline. The backend will invoke **{selected_model}** to autogenerate an ASC 326 compliance report utilizing localized portfolio metrics and algorithm metadata.")
    
    if st.button("Generate MRM Audit Report", use_container_width=True, type="primary"):
        with st.spinner(f"Orchestrating {selected_model} for Governance Reporting..."):
            try:
                res = requests.post(f"{API_URL}/generate_mrm_report", json={"provider": selected_model})
                if res.status_code == 200:
                    st.info(res.json()['report'])
                else:
                    # Professional error handling instructing user to switch model
                    st.error(f"Provider Error: The LLM provider returned a non-200 status code.")
                    st.warning("💡 **Tip:** The selected AI provider might be experiencing temporary high traffic or rate limits. Please select a different LLM provider from the dropdown above and try again!")
            except Exception as e:
                st.error(f"Backend Connection Error: {e}")
