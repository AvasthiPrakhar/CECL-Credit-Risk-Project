# 🏦 CECL Credit Risk Assessment & AI Governance Platform

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-150458?style=for-the-badge&logo=xgboost&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

### 🔗 Quick Links
*   **🔴 Live Application:** [Click here to view the live Dashboard](https://prakhar-cecl-risk.streamlit.app/)
*   **💼 LinkedIn:** [Prakhar Avasthi](http://www.linkedin.com/in/prakhar-avasthi-35067a1bb)
*   **📧 Email:** prakharavasthi1999@gmail.com
*   **📊 Kaggle:** [Prakhar Avasthi](https://www.kaggle.com/avasthiprakhar)

---

## 📌 Project Overview
This project is an end-to-end, industrial-grade **Machine Learning and Generative AI microservice**. It is designed to solve a critical regulatory challenge in quantitative finance: compliance with the **FASB ASC 326 (CECL)** framework.

The platform utilizes a dual-model machine learning architecture to forecast lifetime Expected Credit Losses (ECL), and orchestrates Large Language Models (LLMs) via LangChain to dynamically auto-generate Model Risk Management (MRM) audit reports.

## 🏗️ Microservices Architecture
Unlike standard monolithic data science scripts, this project is built using a decoupled, production-ready architecture:

1. **Backend (FastAPI):** A scaled-to-zero REST API hosted on Render. It handles all heavy ML inference (XGBoost/Random Forest), data scaling, and LLM API orchestration. 
2. **Frontend (Streamlit):** A lightweight, interactive UI hosted on Streamlit Community Cloud. It routes HTTP requests to the backend API, ensuring the presentation layer is entirely decoupled from the computational layer.

## 🧠 Machine Learning Pipeline
The pipeline was trained on **~1.34 Million** historical loan records from a peer-to-peer lending platform. 

*   **Class Imbalance Handling:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the models accurately learned the mathematical boundaries of default risk without majority-class bias.
*   **Probability of Default (PD):** An **XGBoost Classifier** predicts the binary likelihood of default, effectively capturing non-linear feature interactions (e.g., Debt-to-Income vs. Annual Income). *(Hold-out ROC-AUC: 0.81 | Recall: 88%)*
*   **Loss Given Default (LGD):** A **Random Forest Regressor** trained strictly on defaulted loans predicts the percentage of exposure permanently lost after recoveries. *(Hold-out R²: 0.74)*
*   **CECL Scenario Modeling:** Calculates probability-weighted expected losses across Base (50%), Pessimistic (30%), and Optimistic (20%) macroeconomic scenarios.

## 🤖 Dynamic AI Governance (LLM Orchestration)
Model Risk Management (MRM) is traditionally a massive bottleneck in banking. This platform features **Dynamic Service Discovery**:
*   The FastAPI backend actively queries the **Groq API** to fetch a live list of supported LLMs (e.g., LLaMA 3.1, LLaMA 3.3).
*   The frontend dynamically populates a selection menu, allowing the user to route the query to their preferred AI architecture.
*   The backend injects specific loan parameters, model performance metrics, and methodological context into the LLM, auto-generating a highly professional, customized audit compliance report.
*   Features native **SMTP Email Integration** to draft and deliver HTML-formatted assessment reports directly to stakeholders.

