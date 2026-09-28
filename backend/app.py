"""
Streamlit Frontend Interface for CECL Credit Risk Assessment.
"""
import os
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Prakhar Avasthi | AI & Risk", page_icon="🏦")

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR BRANDING
# ==========================================
with st.sidebar:
    st.title("Prakhar Avasthi")
    st.markdown("**Data Science & AI Professional**")
    st.divider()
    
    st.link_button("🔗 LinkedIn", "https://linkedin.com/in/your-profile", use_container_width=True)
    st.link_button("🐙 GitHub", "https://github.com/your-github", use_container_width=True)
    st.link_button("📊 Kaggle", "https://kaggle.com/your-kaggle", use_container_width=True)
    
    st.divider()
    st.markdown("📧 prakharavasthi1999@gmail.com")
    st.caption("Powered by FastAPI, XGBoost, and Groq/Google LLMs.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)
def fetch_available_models():
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

def send_direct_email(to_email, subject, body):
    """Sends an email directly via SMTP using Streamlit Secrets."""
    # These secrets must be configured in Streamlit Cloud!
    sender_email = st.secrets.get("SMTP_EMAIL")
    sender_password = st.secrets.get("SMTP_PASSWORD")
    
    if not sender_email or not sender_password:
        return False, "SMTP credentials missing. Developer needs to set SMTP_EMAIL and SMTP_PASSWORD in Streamlit Secrets."
        
    try:
        msg = MIMEMultipart()
        msg['From'] = f"Prakhar's AI Risk Platform <{sender_email}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# ==========================================
# TABS
# ==========================================
tab_dashboard, tab_methodology = st.tabs(["🏦 Risk & AI Dashboard", "🧠 Model Methodology"])

with tab_dashboard:
    st.title("🏦 CECL Credit Risk & AI Governance Dashboard")
    st.divider()

    available_models = fetch_available_models()
    model_map = {m["display_name"]: m["id"] for m in available_models}

    col1, col2 = st.columns(2, gap="large")

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
            # Save parameters to session state so the Report generator can see them
            st.session_state['current_params'] = payload
            
            with st.spinner("Executing Inference via FastAPI Backend..."):
                try:
                    res = requests.post(f"{API_URL}/predict_cecl", json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state['current_results'] = data
                        st.success("Inference Complete.")
                        
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
            selected_display_name = st.selectbox("Select Target LLM Architecture", options=list(model_map.keys()))
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh API Models"):
                st.cache_data.clear()
                st.rerun()
                
        target_model_id = model_map[selected_display_name]
        st.markdown(f"The backend will invoke **{target_model_id}** to autogenerate a personalized compliance report based on the specific loan parameters entered.")
        
        if st.button("Generate MRM Audit Report", use_container_width=True, type="primary"):
            if 'current_params' not in st.session_state:
                st.warning("⚠️ Please run the Prediction (Step 1) first so the AI has data to analyze!")
            else:
                with st.spinner(f"Orchestrating {target_model_id}..."):
                    try:
                        req_payload = {
                            "model_id": target_model_id,
                            "loan_params": st.session_state['current_params'],
                            "prediction_results": st.session_state['current_results']
                        }
                        res = requests.post(f"{API_URL}/generate_mrm_report", json=req_payload)
                        if res.status_code == 200:
                            st.session_state['generated_report'] = res.json()['report']
                        else:
                            st.error(f"Provider Error: {res.json().get('detail', 'Unknown Error')}")
                    except Exception as e:
                        st.error(f"Backend Connection Error: {e}")
        
        # Display Report & DIRECT Email Options
        if 'generated_report' in st.session_state:
            with st.container(border=True):
                st.markdown(st.session_state['generated_report'])
            
            st.markdown("### 📩 Email Assessment Directly")
            st.markdown("Enter an email below to send the CECL metrics, loan parameters, and the AI audit report directly to their inbox.")
            
            email_target = st.text_input("Recipient Email Address:", placeholder="manager@company.com")
            
            if st.button("Send Direct Email ✉️", use_container_width=True):
                if not email_target:
                    st.warning("Please enter a valid email address.")
                else:
                    with st.spinner("Sending email securely..."):
                        # Combine Parameters, Results, and Report into one clean email body
                        formatted_params = json.dumps(st.session_state['current_params'], indent=4)
                        formatted_results = json.dumps(st.session_state['current_results'], indent=4)
                        
                        full_email_body = f"""
CECL Risk Assessment & AI Audit
====================================

LOAN PARAMETERS SUBMITTED:
{formatted_params}

MODEL PREDICTION (CECL METRICS):
{formatted_results}

AI MODEL GOVERNANCE REPORT:
{st.session_state['generated_report']}

------------------------------------
Generated by Prakhar Avasthi's AI Risk Platform.
"""
                        success, message = send_direct_email(
                            to_email=email_target, 
                            subject="CECL Risk Assessment & AI Audit", 
                            body=full_email_body
                        )
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)


with tab_methodology:
    st.title("🧠 Predictive Modeling & AI Architecture")
    st.divider()
    
    st.subheader("📊 Evaluation Metrics (Hold-out Test Set)")
    met1, met2, met3, met4 = st.columns(4)
    met1.metric(label="PD ROC-AUC", value="0.81")
    met2.metric(label="PD Recall (Default Capture)", value="88%")
    met3.metric(label="LGD R² Score", value="0.74")
    met4.metric(label="LGD RMSE", value="0.18")
    
    st.divider()

    col_m1, col_m2 = st.columns(2, gap="large")
    with col_m1:
        st.subheader("1. Data Processing & Class Imbalance")
        st.write("The pipeline ingested **~1.34 Million historical loan records**.")
        st.info("**Solution (SMOTE):** Synthetic Minority Over-sampling Technique (SMOTE) was applied to oversample the defaulted loans in the training set.")
        st.subheader("3. Expected Credit Loss (CECL) & Scenarios")
        st.markdown("""
        * **Base Scenario (50% weight):** Normal economic conditions.
        * **Pessimistic (30% weight):** Modeled economic downturn.
        * **Optimistic (20% weight):** Modeled economic boom.
        """)
        
    with col_m2:
        st.subheader("2. Dual-Model Architecture")
        with st.expander("Probability of Default (PD) -> XGBoost", expanded=True):
            st.markdown("* `n_estimators`: 100\n* `max_depth`: 4\n* `learning_rate`: 0.1")
        with st.expander("Loss Given Default (LGD) -> Random Forest", expanded=True):
            st.markdown("* `n_estimators`: 30\n* `max_depth`: 10")
        st.subheader("4. Generative AI Governance (RAG)")
        st.write("This application dynamically feeds the portfolio metrics, inference results, and pipeline architecture into an LLM orchestrator to auto-generate audit-ready governance documentation.")
