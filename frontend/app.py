"""
Streamlit Frontend Interface for CECL Credit Risk Assessment.
"""
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(layout="wide", page_title="CECL & RAG Dashboard")
st.title("🏦 CECL Credit Risk & AI Governance Dashboard")

st.markdown("""
*This application implements a decoupled microservices architecture. The Streamlit frontend routes HTTP requests to a FastAPI backend hosting XGBoost/Random Forest models.*
""")
st.divider()

# Cache the API call so we don't bombard Groq's servers every time a user clicks a button
@st.cache_data(ttl=3600)
def fetch_available_models():
    """Fetches dynamic model list from backend."""
    try:
        # INCREASED TIMEOUT TO 90s to allow Render backend to cold-start!
        res = requests.get(f"{API_URL}/models", timeout=90)
        if res.status_code == 200:
            return res.json().get("models", [])
    except Exception as e:
        st.warning(f"Could not fetch live models. Backend might still be waking up. (Error: {e})")
    
    # Updated Fallback list for current APIs
    return [
        {"id": "llama-3.1-8b-instant", "display_name": "Groq (llama-3.1-8b-instant) - Fallback"},
        {"id": "gemini-3.8-flash", "display_name": "Google (gemini-3.8-flash) - Fallback"}
    ]

available_models = fetch_available_models()
# Create a dictionary to map the display name to the actual model ID
model_map = {m["display_name"]: m["id"] for m in available_models}

col1, col2 = st.columns(2)

with col1:
    st.header("1. Loan Parameter Configuration")
    
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
    
    # NEW: A button to manually clear the cache and refetch models if it timed out!
    if st.button("🔄 Refresh API Models", help="Click this to wake up the backend and fetch live models from Groq."):
        st.cache_data.clear()
        st.rerun()

    # User selects display name, we capture it
    selected_display_name = st.selectbox(
        "Select Target LLM Architecture (Dynamically Fetched)", 
        options=list(model_map.keys()),
        help="This list is dynamically populated from the active APIs via Service Discovery."
    )
    
    # Translate display name to the exact API Model ID
    target_model_id = model_map[selected_display_name]
    
    st.markdown(f"Initialize pipeline. The backend will invoke **{target_model_id}** to autogenerate an ASC 326 compliance report utilizing localized portfolio metrics and algorithm metadata.")
    
    if st.button("Generate MRM Audit Report", use_container_width=True, type="primary"):
        with st.spinner(f"Orchestrating {target_model_id} for Governance Reporting..."):
            try:
                # We send the exact ID to the backend
                res = requests.post(f"{API_URL}/generate_mrm_report", json={"model_id": target_model_id})
                
                if res.status_code == 200:
                    st.info(res.json()['report'])
                else:
                    st.error(f"Provider Error: {res.json().get('detail', 'Unknown Error')}")
                    st.warning("💡 **Tip:** Check if your Groq API key has access to this specific model, or try selecting a different one.")
            except Exception as e:
                st.error(f"Backend Connection Error: {e}")
