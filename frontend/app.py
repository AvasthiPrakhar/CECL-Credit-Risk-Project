import streamlit as st
import requests
import os

# Fetch Backend API URL from environment variables (Defaults to local server for testing)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(layout="wide", page_title="CECL & RAG Dashboard")
st.title("🏦 CECL Credit Risk & AI Governance Dashboard")

st.markdown("""
*This application uses a microservices architecture. A Streamlit frontend communicates with a decoupled FastAPI backend hosting XGBoost/Random Forest models.*
**Note:** The API is hosted on a free cloud tier. The very first request may take ~50 seconds to wake the server up. Subsequent requests are instant.
""")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("1. Input Loan Parameters")
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
        
        with st.spinner("Connecting to FastAPI Backend..."):
            try:
                res = requests.post(f"{API_URL}/predict_cecl", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.success("Prediction Complete!")
                    st.metric("Probability of Default (PD)", f"{data['PD']*100:.2f}%")
                    st.metric("Loss Given Default (LGD)", f"{data['LGD']*100:.2f}%")
                    st.metric("Total Expected Loss (CECL)", f"${data['Expected_Credit_Loss']:,.2f}")
                else:
                    st.error(f"API Error: {res.status_code} - {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to backend. Is the FastAPI server running?")

with col2:
    st.header("2. AI Model Risk Governance")
    st.markdown("Click below to trigger the RAG pipeline. The backend will invoke Google Gemini to generate an ASC 326 compliance report based on the underlying code and portfolio metrics.")
    
    if st.button("Generate MRM Audit Report", use_container_width=True, type="primary"):
        with st.spinner("FastAPI is invoking LLM for Governance Report..."):
            try:
                res = requests.post(f"{API_URL}/generate_mrm_report")
                if res.status_code == 200:
                    st.info(res.json()['report'])
                else:
                    st.error(f"API Error: {res.status_code} - {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to backend. Is the FastAPI server running?")
