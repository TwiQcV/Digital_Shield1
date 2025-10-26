import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Digital Shield AI Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #fdf6e6;
        }
        .main-title {
            font-size: 3rem;
            font-weight: bold;
            color: #fdf6e6;
            text-align: center;
            margin-top: 40px;
        }
        .sub-title {
            font-size: 1.4rem;
            text-align: center;
            color: #9aa0a6;
            margin-bottom: 20px;
        }
        .intro-text {
            text-align: center;
            color: #c9c9c9;
            font-size: 1.05rem;
            margin-bottom: 60px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        .card {
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 0 25px rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
            height: 230px;
            color: #fdf6e6;
        }
        .card:hover {
            transform: translateY(-8px);
            box-shadow: 0 0 35px rgba(255, 255, 255, 0.15);
        }
        .card-title {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .card-icon {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        /* ألوان مختلفة لكل كرت */
        .card-blue {
            background: linear-gradient(135deg, #1f3b73, #2a5298);
        }
        .card-cyan {
            background: linear-gradient(135deg, #136a8a, #267871);
        }
        .card-purple {
            background: linear-gradient(135deg, #42275a, #734b6d);
        }
        /* الزر */
        .start-btn {
            display: inline-block;
            margin: 60px auto 0;
            background: linear-gradient(135deg, #fdf6e6, #d6b676);
            color: #1a2946;
            font-size: 1.1rem;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 25px;
            text-align: center;
            transition: 0.3s;
            text-decoration: none;
        }
        .start-btn:hover {
            background: linear-gradient(135deg, #ffecc2, #f7d27c);
            color: #000;
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 70px;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<div class='main-title'>🛡️ Digital Shield AI Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Powered Financial Risk & Cybersecurity Intelligence</div>", unsafe_allow_html=True)

# ---------- INTRO TEXT ----------
st.markdown("""
<div class='intro-text'>
Welcome to <b>Digital Shield</b> — your intelligent dashboard for analyzing cybersecurity data and predicting potential financial losses.
Harness the power of machine learning and AI chat assistance to make informed, data-driven security decisions.
</div>
""", unsafe_allow_html=True)

# ---------- THREE FEATURE CARDS ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class='card card-blue'>
            <div class='card-icon'>💹</div>
            <div class='card-title'>Financial Loss Model</div>
            <p>Predict potential monetary loss from cybersecurity incidents with precision.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='card card-cyan'>
            <div class='card-icon'>🤖</div>
            <div class='card-title'>RAG Chatbot</div>
            <p>Chat with an intelligent assistant to gain instant insights and explanations.</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class='card card-purple'>
            <div class='card-icon'>ℹ️</div>
            <div class='card-title'>Information</div>
            <p>Access data summaries, reports, and model insights for better understanding.</p>
        </div>
    """, unsafe_allow_html=True)

# ---------- SMALL START BUTTON ----------
st.markdown("<div style='text-align:center;'><a class='start-btn' href='#'>🚀 Start</a></div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<div class='footer'>© 2025 Digital Shield Project — Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)

#Home page end-----------------

# # Financial Loss Predictor - Single Page
# import streamlit as st
# import pandas as pd
# from pathlib import Path
# import sys

# # ---------- ADD PROJECT ROOT TO PYTHON PATH ----------
# sys.path.append(str(Path(__file__).resolve().parents[1]))

# # ---------- IMPORT MODEL MODULES ----------
# from Digital_Shield_Packages.ML.fanancial_loss_model import FinancialLossPipeline, DataPreprocessor, FeatureEngineer

# # ---------- PAGE CONFIG ----------
# st.set_page_config(
#     page_title="Digital Shield AI - Financial Loss Predictor",
#     page_icon="💰",
#     layout="wide",
# )

# # ---------- CUSTOM CSS ----------
# st.markdown("""
# <style>
# body { background-color: #0e1117; color: #fdf6e6; }
# .main-title { font-size: 2.8rem; font-weight: bold; color: #fdf6e6; text-align: center; margin-top: 30px; }
# .sub-title { font-size: 1.3rem; text-align: center; color: #9aa0a6; margin-bottom: 40px; }
# .card { background: linear-gradient(135deg, #1a2946 0%, #22335e 100%); border-radius: 20px; padding: 20px; box-shadow: 0 0 20px rgba(255, 255, 255, 0.05); transition: all 0.3s ease; }
# .card:hover { transform: translateY(-5px); box-shadow: 0 0 30px rgba(255, 255, 255, 0.1); }
# .card-title { font-size: 1.2rem; color: #fdf6e6; font-weight: bold; margin-bottom: 10px; }
# .start-btn { display: inline-block; background-color: #4c6ef5; color: #fff; font-size: 1rem; font-weight: bold; border-radius: 10px; padding: 8px 20px; text-align: center; transition: 0.3s; margin-top: 20px; }
# .start-btn:hover { background-color: #3b5bdb; color: #fff; }
# .footer { text-align: center; color: #666; margin-top: 60px; font-size: 0.9rem; }
# </style>
# """, unsafe_allow_html=True)

# # ---------- HEADER ----------
# st.markdown("<div class='main-title'>💰 Financial Loss Predictor</div>", unsafe_allow_html=True)
# st.markdown("<div class='sub-title'>Predict potential financial loss based on cybersecurity incident features</div>", unsafe_allow_html=True)

# # ---------- INSTRUCTIONS ----------
# st.markdown("""
# ### 📝 How to use
# - Fill in the incident details in the form below.
# - Click **Predict** to see the estimated financial loss.
# - All inputs are used by the AI model to generate a prediction.
# """)
# # ---------- DYNAMIC MODEL LOADER ----------
# import os

# # List of possible paths where the model might be
# possible_paths = [
#     "models/financial_loss_xgboost.pkl",                      # default folder
#     "Digital_Shield_Packages/ML/models/financial_loss_xgboost.pkl",  # alternative folder
# ]

# MODEL_PATH = None
# for path in possible_paths:
#     if os.path.exists(path):
#         MODEL_PATH = path
#         break

# if MODEL_PATH is None:
#     st.error(
#         "⚠ Financial loss model not found.\n"
#         f"Tried paths:\n- {possible_paths[0]}\n- {possible_paths[1]}\n"
#         "Please ensure the model is trained and placed in one of these locations."
#     )
#     st.stop()

# # ---------- LOAD THE MODEL ----------
# pipeline = FinancialLossPipeline(model_save_path=MODEL_PATH)

# # ---------- LOAD MODEL ----------
# MODEL_PATH = "models/financial_loss_xgboost.pkl"
# pipeline = FinancialLossPipeline(model_save_path=MODEL_PATH)

# if not Path(MODEL_PATH).exists():
#     st.error(f"Financial loss model not found at {MODEL_PATH}. Train the model first.")
#     st.stop()

# # ---------- FORM INPUTS ----------
# st.markdown("### ⚙️ Incident Details")
# with st.form(key="financial_loss_form"):
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         number_of_users = st.number_input("Number of affected users", min_value=0, value=10)
#         year = st.number_input("Year of incident", min_value=2000, max_value=2030, value=2025)
#         incident_resolution_hours = st.number_input("Incident resolution time (hours)", min_value=0.0, value=24.0)

#     with col2:
#         data_breach_gb = st.number_input("Data breach in GB", min_value=0.0, value=5.0)

#     with col3:
#         organization_type = st.selectbox("Organization Type", ["Healthcare", "Banking", "Education", "Other"])
#         incident_type = st.selectbox("Incident Type", ["Malware", "Phishing", "Ransomware", "Other"])

#     submit_btn = st.form_submit_button("Predict")

# # ---------- MAKE PREDICTION ----------
# if submit_btn:
#     input_dict = {
#         "number of affected users": [number_of_users],
#         "year": [year],
#         "incident resolution time (in hours)": [incident_resolution_hours],
#         "data breach in gb": [data_breach_gb],
#         "organization type": [organization_type],
#         "incident type": [incident_type]
#     }
#     X_new = pd.DataFrame(input_dict)
#     X_new_clean, _ = DataPreprocessor.clean_data(X_new, pd.Series([0]*len(X_new)))
#     X_new_features = FeatureEngineer.engineer_features(X_new_clean)

#     prediction = pipeline.predict(X_new_features)[0]

#     st.markdown(f"### 💵 Estimated Financial Loss: **${prediction:,.2f} Million**")

# # ---------- FOOTER ----------
# st.markdown("<div class='footer'>© 2025 Digital Shield Project — Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)
# end page--------------------


# ---------- INFORMATION SECTION (New Added Part) ----------
st.markdown("<hr style='margin-top:60px;margin-bottom:40px;border:1px solid #444;'>", unsafe_allow_html=True)

st.markdown("<div class='main-title'>ℹ️ Cybersecurity Information Center</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Key terms and insights from your dataset</div>", unsafe_allow_html=True)

# ---------- INTRO ----------
st.markdown("""
<div class='intro-text'>
Here you can explore brief and clear explanations of the core cybersecurity features in your dataset — including Attack Types, Defense Mechanisms, and Security Vulnerability Types.
</div>
""", unsafe_allow_html=True)

# ---------- ATTACK TYPES ----------
st.markdown("## 🛠️ Attack Types Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='card card-blue'>
        <div class='card-title'>Phishing</div>
        <p>A social engineering attack that tricks users into revealing sensitive information via fake emails or websites.</p>
    </div>
    <div class='card card-cyan' style='margin-top:20px;'>
        <div class='card-title'>Ransomware</div>
        <p>Malicious software that encrypts files and demands payment to restore access.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='card card-purple'>
        <div class='card-title'>DDoS (Distributed Denial of Service)</div>
        <p>An attack that floods a network or server with traffic, making it unavailable to legitimate users.</p>
    </div>
    <div class='card card-blue' style='margin-top:20px;'>
        <div class='card-title'>Malware</div>
        <p>Any software intentionally designed to cause damage, steal data, or disrupt systems.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='card card-cyan'>
        <div class='card-title'>SQL Injection</div>
        <p>Exploits vulnerabilities in database queries to access or modify data.</p>
    </div>
    <div class='card card-purple' style='margin-top:20px;'>
        <div class='card-title'>Man-in-the-Middle (MitM)</div>
        <p>Intercepts and manipulates communication between two parties without their knowledge.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- DEFENSE MECHANISMS ----------
st.markdown("<br><br><div class='main-title' style='font-size:2.2rem;'>🧰 Defense Mechanisms</div>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class='card card-blue'>
        <div class='card-title'>Firewall</div>
        <p>Monitors and filters incoming and outgoing network traffic based on security rules.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class='card card-cyan'>
        <div class='card-title'>Intrusion Detection System (IDS)</div>
        <p>Detects suspicious activities and potential threats within a network or system.</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class='card card-purple'>
        <div class='card-title'>Encryption</div>
        <p>Secures data by converting it into an unreadable format, accessible only with a decryption key.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- SECURITY VULNERABILITY TYPES ----------
st.markdown("<br><br><div class='main-title' style='font-size:2.2rem;'>🧩 Security Vulnerability Types</div>", unsafe_allow_html=True)

col7, col8, col9 = st.columns(3)

with col7:
    st.markdown("""
    <div class='card card-blue'>
        <div class='card-title'>Zero-Day Vulnerability</div>
        <p>A software flaw exploited by attackers before the developer becomes aware of it.</p>
    </div>
    """, unsafe_allow_html=True)

with col8:
    st.markdown("""
    <div class='card card-cyan'>
        <div class='card-title'>Weak Authentication</div>
        <p>Occurs when poor or reused passwords make it easy for attackers to gain access.</p>
    </div>
    """, unsafe_allow_html=True)

with col9:
    st.markdown("""
    <div class='card card-purple'>
        <div class='card-title'>Unpatched Software</div>
        <p>Software that has not been updated, leaving it exposed to known security vulnerabilities.</p>
    </div>
    """, unsafe_allow_html=True)
