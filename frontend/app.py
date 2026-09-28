"""
Streamlit Frontend Interface for CECL Credit Risk Assessment.
"""
import os
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import urllib.parse
import markdown
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Prakhar Avasthi | AI & Risk", page_icon="🏦")

# Advanced Custom CSS to increase Font Sizes and aesthetic appeal
st.markdown("""
    <style>
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Increase Font Size for the Navigation Tabs */
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 22px !important;
        font-weight: 600 !important;
    }
    
    /* Email Draft Button Styling */
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
# 2. SIDEBAR BRANDING & SOCIAL LINKS
# ==========================================
with st.sidebar:
    st.markdown("## Prakhar Avasthi")
    st.markdown("#### Data Science & AI Professional")
    
    st.divider()
    
    st.link_button("🔗 LinkedIn", "http://www.linkedin.com/in/prakhar-avasthi-35067a1bb", use_container_width=True)
    st.link_button("🐙 GitHub", "https://github.com/AvasthiPrakhar", use_container_width=True)
    st.link_button("📊 Kaggle", "https://www.kaggle.com/avasthiprakhar", use_container_width=True)
    
    st.divider()
    st.markdown("📧 **prakharavasthi1999@gmail.com**")
    st.caption("Powered by FastAPI, XGBoost, and Groq/Google LLMs.")

# ==========================================
# 3. HELPER FUNCTIONS (Including HTML Email)
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
        {"id": "llama-3.3-70b-versatile", "display_name": "Groq (llama-3.3-70b-versatile)"},
        {"id": "gemini-3.8-flash", "display_name": "Google (gemini-3.8-flash)"}
    ]

def send_html_email(to_email, subject, params_dict, metrics_dict, raw_markdown_report):
    """Constructs and sends a beautifully formatted HTML email."""
    sender_email = st.secrets.get("SMTP_EMAIL")
    sender_password = st.secrets.get("SMTP_PASSWORD")
    
    if not sender_email or not sender_password:
        return False, "SMTP credentials missing in Streamlit Secrets."
        
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = f"Prakhar's AI Risk Platform <{sender_email}>"
        msg['To'] = to_email
        msg['Subject'] = subject

        # Convert LLM Markdown report to clean HTML
        report_html = markdown.markdown(raw_markdown_report)
        
        # Build HTML Tables for parameters and metrics
        params_rows = "".join([f"<tr><td style='padding:8px; border-bottom:1px solid #ddd;'><b>{k}</b></td><td style='padding:8px; border-bottom:1px solid #ddd;'>{v}</td></tr>" for k, v in params_dict.items()])
        
        # Format the CECL Metric specifically as currency
        ecl_formatted = f"${metrics_dict['Expected_Credit_Loss']:,.2f}"
        
        html_content = f"""
        <html>
          <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6; max-width: 800px; margin: auto;">
            <div style="background-color: #1C83E1; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                <h2 style="margin: 0;">CECL Risk Assessment & AI Audit</h2>
            </div>
            
            <div style="padding: 20px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 8px 8px;">
                <h3 style="color: #1C83E1;">1. Applicant Loan Parameters</h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                    {params_rows}
                </table>
                
                <h3 style="color: #1C83E1;">2. Quantitative Inference (Model Output)</h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; background-color: #f9f9f9;">
                    <tr><td style='padding:12px; border-bottom:1px solid #ddd;'><b>Probability of Default (PD):</b></td><td style='padding:12px; border-bottom:1px solid #ddd;'>{metrics_dict['PD']*100:.2f}%</td></tr>
                    <tr><td style='padding:12px; border-bottom:1px solid #ddd;'><b>Loss Given Default (LGD):</b></td><td style='padding:12px; border-bottom:1px solid #ddd;'>{metrics_dict['LGD']*100:.2f}%</td></tr>
                    <tr><td style='padding:12px; border-bottom:1px solid #ddd; color: #d9534f;'><b>Expected Credit Loss (CECL):</b></td><td style='padding:12px; border-bottom:1px solid #ddd; color: #d9534f;'><b>{ecl_formatted}</b></td></tr>
                </table>
                
                <h3 style="color: #1C83E1;">3. AI Model Governance Report</h3>
                <div style="background-color: #f4f6f9; padding: 15px; border-left: 4px solid #1C83E1; border-radius: 4px;">
                    {report_html}
                </div>
                
                <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
                <div style="text-align: center; color: #777; font-size: 12px;">
                    <p>Automated Assessment Generated by <b>Prakhar Avasthi's AI Risk Platform</b></p>
                    <a href="http://www.linkedin.com/in/prakhar-avasthi-35067a1bb" style="color: #1C83E1; text-decoration: none; font-weight: bold;">Connect with Prakhar on LinkedIn</a>
                </div>
            </div>
          </body>
        </html>
        """
        
        msg.attach(MIMEText(html_content, 'html'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, "HTML Email sent successfully!"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# ==========================================
# 4. TOP NAVIGATION BAR (TABS)
# ==========================================
tab_dashboard, tab_methodology = st.tabs(["🏦 Risk & AI Dashboard", "🧠 Model Methodology"])

# ------------------------------------------
# TAB 1: DASHBOARD
# ------------------------------------------
with tab_dashboard:
    st.title("🏦 CECL Credit Risk & AI Governance Dashboard")
    st.markdown("*A decoupled microservices architecture routing HTTP requests to a FastAPI backend hosting ML models.*")
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
        
        if 'generated_report' in st.session_state:
            with st.container(border=True):
                st.markdown(st.session_state['generated_report'])
            
            st.markdown("### 📩 Email Assessment Directly")
            st.markdown("Enter an email below to send the beautifully formatted HTML report (including the metrics and parameters) directly to an inbox.")
            
            email_target = st.text_input("Recipient Email Address:", placeholder="manager@company.com")
            
            if st.button("Send Direct Email ✉️", use_container_width=True):
                if not email_target:
                    st.warning("Please enter a valid email address.")
                else:
                    with st.spinner("Compiling and sending HTML email..."):
                        success, message = send_html_email(
                            to_email=email_target, 
                            subject="CECL Risk Assessment & AI Audit", 
                            params_dict=st.session_state['current_params'],
                            metrics_dict=st.session_state['current_results'],
                            raw_markdown_report=st.session_state['generated_report']
                        )
                        if success:
                            st.success(message)
                            st.balloons()
                        else:
                            st.error(message)


# ------------------------------------------
# TAB 2: METHODOLOGY & METRICS (EXPANDED)
# ------------------------------------------
with tab_methodology:
    st.title("🧠 Predictive Modeling & AI Architecture")
    st.markdown("A deep dive into the machine learning pipeline, econometric methodologies, and ASC 326 compliance frameworks utilized in this application.")
    st.divider()
    
    st.subheader("📊 Evaluation Metrics (Hold-out Test Set)")
    st.markdown("To prevent data leakage and ensure generalization, all models were evaluated against a strict 20% hold-out validation set. The metrics below demonstrate enterprise-grade performance suitable for consumer credit risk assessment.")
    
    met1, met2, met3, met4 = st.columns(4)
    met1.metric(label="PD ROC-AUC", value="0.81")
    met2.metric(label="PD Recall (Default Capture)", value="88%")
    met3.metric(label="LGD R² Score", value="0.74")
    met4.metric(label="LGD RMSE", value="0.18")
    
    st.divider()

    col_m1, col_m2 = st.columns(2, gap="large")
    
    with col_m1:
        st.subheader("1. Data Processing & Class Imbalance")
        st.write("The pipeline ingested **~1.34 Million historical loan records** from a peer-to-peer lending platform. A major challenge in quantitative finance is class imbalance—because most borrowers pay back their loans, default classes are heavily underrepresented.")
        st.info("**Solution (SMOTE):** Rather than simple random oversampling (which leads to overfitting), Synthetic Minority Over-sampling Technique (SMOTE) was applied. SMOTE interpolates between existing minority instances to generate entirely new, realistic default profiles, allowing the algorithms to accurately learn the mathematical boundaries of risk.")
        
        st.subheader("3. Expected Credit Loss (CECL) & Scenarios")
        st.write("Historically, banks used an 'Incurred Loss' model. Under the new **FASB ASC 326 framework (CECL)**, lenders must forecast *lifetime* expected credit losses (ECL = PD × LGD × EAD) incorporating forward-looking macroeconomic data.")
        st.write("To simulate this compliance, this application generates a probability-weighted outcome across three economic vectors:")
        st.markdown("""
        * **Base Scenario (50% weight):** Assumes standard, stable economic conditions.
        * **Pessimistic Scenario (30% weight):** Simulates an economic downturn, inflating PD by 20% and LGD by 10%.
        * **Optimistic Scenario (20% weight):** Simulates an economic boom, reducing PD by 20% and LGD by 10%.
        """)
        
    with col_m2:
        st.subheader("2. Dual-Model ML Architecture")
        
        with st.expander("Probability of Default (PD) → XGBoost", expanded=True):
            st.write("An Extreme Gradient Boosting (XGBoost) classifier was utilized to predict the binary likelihood of default. XGBoost was selected for its exceptional ability to handle non-linear feature interactions (like Debt-to-Income vs. Annual Income) and its robustness to outliers.")
            st.markdown("""
            **Optimized Hyperparameters:**
            * `n_estimators`: 100
            * `max_depth`: 4
            * `learning_rate`: 0.1
            """)
            
        with st.expander("Loss Given Default (LGD) → Random Forest", expanded=True):
            st.write("A Random Forest Regressor was trained *strictly* on defaulted loans to predict the exact percentage of exposure that would be permanently lost (shortfall after recoveries). Random Forest was chosen because its ensemble averaging heavily reduces the variance caused by highly noisy recovery data.")
            st.markdown("""
            **Optimized Hyperparameters:**
            * `n_estimators`: 30
            * `max_depth`: 10
            """)

        st.subheader("4. Generative AI Governance (RAG)")
        st.write("Model Risk Management (MRM) is traditionally a massive bottleneck in banking. This application demonstrates a modern AI integration where dynamic portfolio metrics, inference results, and model methodologies are injected directly into the context window of an LLM (Groq LLaMA or Google Gemini). This allows the system to auto-generate highly accurate, audit-ready governance documentation instantly, completely eliminating human reporting overhead.")
