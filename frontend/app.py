"""
Streamlit Frontend Interface for CECL Credit Risk Assessment.
"""
import os
import requests
import urllib.parse
import streamlit as st

# Define Backend Service Endpoint
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS (BRANDING)
# ==========================================

st.set_page_config(layout="wide", page_title="Prakhar Avasthi | AI & Risk", page_icon="🏦")

# Custom CSS for aesthetic improvements
st.markdown("""
    <style>
    /* Styling the metric cards */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Styling the email button to match Streamlit */
    .email-btn {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #FF4B4B;
        color: white !important;
        text-decoration: none;
        border-radius: 0.5rem;
        font-weight: 600;
        text-align: center;
        width: 100%;
        transition: background-color 0.3s;
    }
    .email-btn:hover {
        background-color: #FF6666;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR NAVIGATION & BRANDING
# ==========================================
with st.sidebar:
    st.title("Prakhar Avasthi")
    st.markdown("**Data Scientist & AI Engineer**")
    
    # Social Links using Markdown
    st.markdown("""
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](http://www.linkedin.com/in/prakhar-avasthi-35067a1bb)
    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AvasthiPrakhar)
    [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/avasthiprakhar)
    """)
    st.markdown("📧 prakharavasthi1999@gmail.com")
    
    st.divider()
    
    # Page Navigation
    page = st.radio("📌 Navigation", ["🏦 Risk & AI Dashboard", "🧠 Model Methodology"])
    
    st.divider()
    st.caption("Powered by FastAPI, XGBoost, and Groq/Google LLMs.")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)
def fetch_available_models():
    """Fetches dynamic model list from backend."""
    try:
        res = requests.get(f"{API_URL}/models", timeout=90)
        if res.status_code == 200:
            return res.json().get("models", [])
    except Exception:
        pass
    return [
        {"id": "llama-3.1-8b-instant", "display_name": "Groq (llama-3.1-8b-instant)"},
        {"id": "gemini-3.8-flash", "display_name": "Google (gemini-3.8-flash)"}
    ]

# ==========================================
# 4. PAGE 1: DASHBOARD
# ==========================================
if page == "🏦 Risk & AI Dashboard":
    st.title("🏦 CECL Credit Risk & AI Governance Dashboard")
    st.markdown("*A decoupled microservices architecture routing HTTP requests to a FastAPI backend hosting XGBoost/Random Forest models.*")
    st.divider()

    available_models = fetch_available_models()
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
                        
                        # Displaying aesthetic metric cards
                        m1, m2, m3 = st.columns(3)
                        m1.metric("PD (Probability of Default)", f"{data['PD']*100:.2f}%")
                        m2.metric("LGD (Loss Given Default)", f"{data['LGD']*100:.2f}%")
                        m3.metric("Total CECL (Expected Loss)", f"${data['Expected_Credit_Loss']:,.2f}")
                    else:
                        st.error(f"Service Error: {res.status_code}")
                except Exception as e:
                    st.error(f"Backend Connection Error: {e}")

    with col2:
        st.header("2. AI Model Risk Governance (MRM)")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            selected_display_name = st.selectbox(
                "Select Target LLM Architecture (Dynamically Fetched)", 
                options=list(model_map.keys())
            )
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh", help="Wake backend and fetch live models"):
                st.cache_data.clear()
                st.rerun()
                
        target_model_id = model_map[selected_display_name]
        st.markdown(f"The backend will invoke **{target_model_id}** to autogenerate an ASC 326 compliance report.")
        
        if st.button("Generate MRM Audit Report", use_container_width=True, type="primary"):
            with st.spinner(f"Orchestrating {target_model_id} for Governance Reporting..."):
                try:
                    res = requests.post(f"{API_URL}/generate_mrm_report", json={"model_id": target_model_id})
                    if res.status_code == 200:
                        report_text = res.json()['report']
                        st.session_state['generated_report'] = report_text # Save to session state
                    else:
                        st.error("Provider Error. Please try a different model.")
                except Exception as e:
                    st.error(f"Backend Connection Error: {e}")
        
        # Display Report & Export Options if generated
        if 'generated_report' in st.session_state:
            st.success("Report Generated Successfully!")
            
            # Using st.code provides a native "Copy to Clipboard" button in the top right!
            st.code(st.session_state['generated_report'], language="markdown")
            
            st.markdown("### Export Report")
            email_target = st.text_input("Enter recipient email address:")
            if email_target:
                # Generate Mailto link
                subject = urllib.parse.quote("CECL MRM Audit Report")
                body = urllib.parse.quote(st.session_state['generated_report'])
                mailto_url = f"mailto:{email_target}?subject={subject}&body={body}"
                
                st.markdown(f'<a href="{mailto_url}" class="email-btn" target="_blank">✉️ Draft Email in Default Client</a>', unsafe_allow_html=True)


# ==========================================
# 5. PAGE 2: METHODOLOGY & METRICS
# ==========================================
elif page == "🧠 Model Methodology":
    st.title("🧠 Predictive Modeling & AI Architecture")
    st.markdown("An overview of the machine learning pipeline and econometric methodologies utilized in this application.")
    
    st.divider()
    
    try:
        metrics_data = requests.get(f"{API_URL}/metrics").json()
        rows_processed = metrics_data.get('Total_Rows_Processed', '1.34 Million')
        exposure = metrics_data.get('Total_Portfolio_Exposure', '$19.3 Billion')
    except:
        rows_processed = "~1.34 Million"
        exposure = "N/A"

    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("1. Data Processing & Class Imbalance")
        st.write(f"The pipeline ingested **{rows_processed} historical loan records**. Because loan defaults are highly imbalanced (representing a minority of cases), standard models often fail to predict them accurately.")
        st.info("**Solution (SMOTE):** Synthetic Minority Over-sampling Technique (SMOTE) was applied to oversample the defaulted loans in the training set, allowing the model to learn the mathematical boundaries of default risk without bias.")
        
        st.subheader("3. Expected Credit Loss (CECL) & Scenarios")
        st.write("In adherence with ASC 326 guidelines, a single Expected Credit Loss (ECL = PD × LGD × EAD) is insufficient. We applied a probability-weighted macroeconomic forecast:")
        st.markdown("- **Base Scenario (50% weight):** Normal economic conditions.")
        st.markdown("- **Pessimistic Scenario (30% weight):** Modeled economic downturn (PD +20%, LGD +10%).")
        st.markdown("- **Optimistic Scenario (20% weight):** Modeled economic boom (PD -20%, LGD -10%).")
        
    with col_m2:
        st.subheader("2. Dual-Model Architecture")
        with st.expander("Probability of Default (PD) -> XGBoost", expanded=True):
            st.write("An Extreme Gradient Boosting (XGBoost) classifier was trained on the balanced dataset to output the mathematical probability that a borrower will default.")
            st.caption("Hyperparameters: n_estimators=100, max_depth=4, learning_rate=0.1")
            
        with st.expander("Loss Given Default (LGD) -> Random Forest", expanded=True):
            st.write("A Random Forest Regressor was trained *strictly* on defaulted loans to predict the recovery rate shortfall. This calculates the exact percentage of the exposure that will be permanently lost.")
            st.caption("Hyperparameters: n_estimators=30, max_depth=10")

        st.subheader("4. Generative AI Governance (RAG)")
        st.write("Model Risk Management (MRM) is traditionally a massive bottleneck in finance. This application dynamically feeds the portfolio metrics and pipeline architecture into an LLM orchestrator (Groq/Google Gemini) to auto-generate audit-ready governance documentation on demand.")
