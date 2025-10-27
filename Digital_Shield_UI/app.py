import streamlit as st
import os
from pathlib import Path

from PIL import Image
import base64
from io import BytesIO

# Load image
project_root = Path(__file__).parent.parent

img = Image.open(project_root / "UI" / "lmags" / "p2.jpg")

# Convert image to base64
buffered = BytesIO()
img.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Display image as HTML banner with fixed height
st.markdown(f"""
<div style="
    width: 100%;
    height: 380px;
    overflow: hidden;
    border-radius: 10px;
    margin-bottom: 20px;
    position: relative;
">
    <img src="data:image/jpeg;base64,{img_str}" style="
        width: 100%;
        height: 100%;
        object-fit: cover;
    ">
</div>
""", unsafe_allow_html=True)

import base64

# project_root = Path(__file__).parent.parent
# gif_path = project_root / "UI" / "lmags" / "v3.gif"

# # Convert GIF to Base64
# with open(gif_path, "rb") as f:
#     data = f.read()
# gif_base64 = base64.b64encode(data).decode()

# # Display GIF with fixed height and rounded corners
# st.markdown(f"""
# <div style="
#     width: 100%;
#     height: 280px;
#     overflow: hidden;
#     border-radius: 10px;
#     margin-bottom: 20px;
# ">
#     <img src="data:image/gif;base64,{gif_base64}" style="
#         width: 100%;
#         height: 100%;
#         object-fit: cover;
#     ">
# </div>
# """, unsafe_allow_html=True)


# # Load local video
# video_path = "/home/nouf_talal/code/TwiQcV/Digital_Shield1/Digital_Shield_Deployment/UI/lmags/v2.mp4"

# # Display video
# st.video(video_path, start_time=0)  #

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

# # ---------- SMALL START BUTTON ----------
# st.markdown("<div style='text-align:center;'><a class='start-btn' href='#'>🚀 Start</a></div>", unsafe_allow_html=True)
# ---------- CUSTOM EXPANDER STYLE ----------
st.markdown("""
    <style>
        /* Remove default expander borders */
        details {
            border: none !important;
        }
        summary::-webkit-details-marker {
            display: none;
        }

        /* --- Base expander style --- */
        details > summary {
            padding: 14px 18px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 1.05rem;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 12px;
            text-align: left;
            color: #fdf6e6;
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.05);
        }

        /* --- Remove the black underline and outline --- */
        details > summary:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        /* --- Individual Colors --- */
        .exp-blue > summary {
            background: linear-gradient(135deg, #1f3b73, #2a5298);
        }
        .exp-blue[open] > summary {
            background: linear-gradient(135deg, #2a5298, #1f3b73);
        }

        .exp-cyan > summary {
            background: linear-gradient(135deg, #136a8a, #267871);
        }
        .exp-cyan[open] > summary {
            background: linear-gradient(135deg, #267871, #136a8a);
        }

        .exp-purple > summary {
            background: linear-gradient(135deg, #42275a, #734b6d);
        }
        .exp-purple[open] > summary {
            background: linear-gradient(135deg, #734b6d, #42275a);
        }

        /* --- Expander hover effect --- */
        details > summary:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
        }

        /* --- Expander content --- */
        details > div {
            background-color: #141a2b;
            border-left: 3px solid rgba(255, 255, 255, 0.1);
            padding: 18px 22px;
            border-radius: 0 0 12px 12px;
            margin-bottom: 18px;
            color: #dcdcdc;
            font-size: 1rem;
            line-height: 1.7;
            box-shadow: 0 0 10px rgba(255,255,255,0.03);
        }

        /* --- Smooth open animation --- */
        details[open] div {
            animation: fadeIn 0.4s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-5px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<div class='footer'>© 2025 Digital Shield Project — Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)

#Home page end-----------------
# ---------- CUSTOM EXPANDER STYLE ----------
st.markdown("""
    <style>
        /* Base expander reset */
        .streamlit-expanderHeader {
            font-weight: bold !important;
            font-size: 1.05rem !important;
            border-radius: 12px !important;
            padding: 12px 18px !important;
            margin-bottom: 10px !important;
            color: #fdf6e6 !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            border: none !important;
        }
        .streamlit-expanderHeader:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.15) !important;
        }
        /* Remove black underline */
        .streamlit-expander {
            border: none !important;
            box-shadow: none !important;
        }
        /* Expander content */
        .streamlit-expanderContent {
            background-color: #141a2b !important;
            border-radius: 0 0 12px 12px !important;
            padding: 18px 22px !important;
            color: #dcdcdc !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            margin-bottom: 20px !important;
            animation: fadeIn 0.4s ease-in-out !important;
        }

        /* Color variants */
        .exp-blue .streamlit-expanderHeader {
            background: linear-gradient(135deg, #1f3b73, #2a5298) !important;
        }
        .exp-cyan .streamlit-expanderHeader {
            background: linear-gradient(135deg, #136a8a, #267871) !important;
        }
        .exp-purple .streamlit-expanderHeader {
            background: linear-gradient(135deg, #42275a, #734b6d) !important;
        }

        /* Animation */
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-5px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)



# ---------- INFORMATION SECTION (New Added Part) ----------
# ---------- INFORMATION SECTION (collapsible cards) ----------
# English comment: This block renders the Information section where each card includes
# a small "collapsible" (st.expander) that reveals concise definition, how it happens,
# and short protection steps. Keep CSS classes (card, card-blue, card-cyan, card-purple).


st.markdown("<hr style='margin-top:60px;margin-bottom:30px;border:1px solid #444;'>", unsafe_allow_html=True)
st.markdown("<div class='main-title'>ℹ️ Cybersecurity Information Center</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Short definitions, how attacks happen, and simple protection steps</div>", unsafe_allow_html=True)

st.markdown("""
<div class='intro-text'>
Quick, practical explanations for common Attack Types, Defense Mechanisms and Vulnerability Types from the dataset.
Click any card's **\"Details\"** to expand concise background and protection advice.
</div>
""", unsafe_allow_html=True)

# --- Attack Types (3 columns layout of preview cards + expanders) ---
st.markdown("### 🛠️ Attack Types")
a1, a2, a3 = st.columns(3)

with a1:
    st.markdown("""
    <div class='card card-blue'>
      <div class='card-title'>Phishing</div>
      <div style='color:#dcdcdc'>Social-engineering messages that trick users.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — Phishing"):
        st.markdown("""
        **Definition:** Deceptive emails/links/messages that trick people into revealing credentials or running malware.
        **How it happens:** Attacker spoofs sender or creates fake sites; user clicks link or enters credentials.
        **Protection (short):**
        - Use email filtering + anti-phishing tools.
        - Enable MFA (multi-factor authentication).
        - Train staff to spot suspicious links and attachments.
        """)

    st.markdown("""<div style='height:10px'></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='card card-cyan'>
      <div class='card-title'>Ransomware</div>
      <div style='color:#dcdcdc'>Malware that encrypts files and demands ransom.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — Ransomware"):
        st.markdown("""
        **Definition:** Malware that encrypts data and extorts payment for decryption.
        **How it happens:** Delivered via phishing, exposed RDP, or malicious downloads.
        **Protection (short):**
        - Regular offline backups; test restores.
        - Patch exposed services and restrict RDP.
        - Use endpoint detection and limit admin privileges.
        """)

with a2:
    st.markdown("""
    <div class='card card-purple'>
      <div class='card-title'>DDoS (Distributed Denial of Service)</div>
      <div style='color:#dcdcdc'>Flooding a target to make it unavailable.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — DDoS"):
        st.markdown("""
        **Definition:** Massive traffic or request floods that overwhelm services or network links.
        **How it happens:** Botnets or spoofed request floods attack bandwidth or resources.
        **Protection (short):**
        - Use upstream DDoS protection / CDN / rate limiting.
        - Design scalable infrastructure and failover routes.
        - Monitor traffic baselines and alert on spikes.
        """)

    st.markdown("""<div style='height:10px'></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='card card-blue'>
      <div class='card-title'>Malware</div>
      <div style='color:#dcdcdc'>General-purpose malicious software (virus, trojan, RAT).</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — Malware"):
        st.markdown("""
        **Definition:** Software designed to harm, steal, or persist (viruses, trojans, remote access tools).
        **How it happens:** Delivered via attachments, downloads, compromised sites or removable media.
        **Protection (short):**
        - Keep endpoints patched and use EDR/AV solutions.
        - Restrict execution policies and remove unnecessary software.
        - Apply least-privilege accounts and network segmentation.
        """)

with a3:
    st.markdown("""
    <div class='card card-cyan'>
      <div class='card-title'>SQL Injection</div>
      <div style='color:#dcdcdc'>Injecting SQL to read/modify databases.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — SQL Injection"):
        st.markdown("""
        **Definition:** Attacker supplies crafted input that alters SQL queries to access or modify data.
        **How it happens:** Poor input validation and string-concatenated queries in apps.
        **Protection (short):**
        - Use parameterized queries / prepared statements.
        - Validate and sanitize user input; apply least privilege for DB accounts.
        - Web application firewall (WAF) and code reviews.
        """)

    st.markdown("""<div style='height:10px'></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='card card-purple'>
      <div class='card-title'>Man-in-the-Middle (MitM)</div>
      <div style='color:#dcdcdc'>Intercepting or altering communications.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — Man-in-the-Middle"):
        st.markdown("""
        **Definition:** Intercepting or changing messages between two parties to eavesdrop or tamper.
        **How it happens:** On insecure Wi-Fi, poorly configured TLS, or compromised routers.
        **Protection (short):**
        - Enforce TLS/HTTPS, use HSTS and certificate validation.
        - Avoid unsecured public Wi-Fi; use VPN for remote access.
        - Implement network segmentation and secure DNS.
        """)

# --- Defense Mechanisms ---
st.markdown("### 🧰 Defense Mechanisms")
d1, d2, d3 = st.columns(3)

with d1:
    st.markdown("""
    <div class='card card-blue'>
      <div class='card-title'>Firewall</div>
      <div style='color:#dcdcdc'>Filters traffic and enforces network rules.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — Firewall"):
        st.markdown("""
        **What it is:** Packet/connection filter that controls allowed traffic.
        **When it helps:** Blocks unwanted ports/services and isolates segments.
        **Quick steps:** Maintain policy, block unused ports, log alerts and combine with IDS.
        """)

with d2:
    st.markdown("""
    <div class='card card-cyan'>
      <div class='card-title'>Intrusion Detection / Prevention (IDS/IPS)</div>
      <div style='color:#dcdcdc'>Detects or blocks suspicious activity on the network.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — IDS/IPS"):
        st.markdown("""
        **What it is:** Monitors traffic or hosts for known bad patterns and alerts or blocks.
        **When it helps:** Early detection of attacks and anomalous behavior.
        **Quick steps:** Tune signatures, feed alerts into SOC workflows, use alongside EDR.
        """)

with d3:
    st.markdown("""
    <div class='card card-purple'>
      <div class='card-title'>Encryption</div>
      <div style='color:#dcdcdc'>Protects data confidentiality at rest and in transit.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — Encryption"):
        st.markdown("""
        **What it is:** Transforming data so only authorized parties can read it (keys).
        **When it helps:** Protects data even if storage or network is compromised.
        **Quick steps:** Use strong algorithms, manage keys securely, encrypt backups.
        """)

# --- Vulnerability Types ---
st.markdown("<🧩 Vulnerability Types")
v1, v2, v3 = st.columns(3)

with v1:
    st.markdown("""
    <div class='card card-blue'>
      <div class='card-title'>Zero-Day</div>
      <div style='color:#dcdcdc'>Vulnerability exploited before a patch exists.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — Zero-Day"):
        st.markdown("""
        **Definition:** A flaw unknown to vendor and unpatched when exploited.
        **Protection (short):**
        - Defense-in-depth (network segmentation, EDR, monitoring).
        - Rapid incident response playbooks and threat intelligence feeds.
        """)

with v2:
    st.markdown("""
    <div class='card card-cyan'>
      <div class='card-title'>Weak Authentication</div>
      <div style='color:#dcdcdc'>Easy-to-guess or reused credentials.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — Weak Authentication"):
        st.markdown("""
        **Definition:** Passwords or auth methods that attackers can easily bypass.
        **Protection (short):**
        - Enforce strong passwords + MFA.
        - Use password managers and block reused/stolen credentials.
        """)

with v3:
    st.markdown("""
    <div class='card card-purple'>
      <div class='card-title'>Unpatched Software</div>
      <div style='color:#dcdcdc'>Known flaws left without vendor fixes applied.</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Details — Unpatched Software"):
        st.markdown("""
        **Definition:** Systems missing vendor updates that fix vulnerabilities.
        **Protection (short):**
        - Maintain a patch schedule, test and deploy quickly.
        - Prioritize internet-facing and high-risk assets.
        """)

# Small footnote and optional preview toggle (keeps layout consistent)
st.markdown("<div style='margin-top:12px;color:#9aa0a6;font-size:0.9rem'>Tip: expand any card for a short, actionable explanation and quick protection steps.</div>", unsafe_allow_html=True)

# (Optional) a checkbox to show a small dataset preview — uncomment if you want:
# if st.checkbox("Show dataset preview (first 50 rows)"):
#     df = pd.read_csv("Digital_Shield_data/proccesed/Cleaned_Digital_Shield_Dataset.csv")
#     st.dataframe(df.head(50))
