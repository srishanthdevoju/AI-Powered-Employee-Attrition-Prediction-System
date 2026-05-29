"""
AI-Powered Employee Attrition Prediction System
Main Streamlit App — Landing / Hero Page
"""

import streamlit as st
import plotly.graph_objects as go

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition AI | Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E1B4B 0%, #0F172A 100%);
    border-right: 1px solid rgba(124, 58, 237, 0.3);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(124, 58, 237, 0.12);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 16px;
    padding: 1rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    border-color: rgba(6, 182, 212, 0.6);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.8);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7C3AED, #06B6D4) !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #06B6D4);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6);
}

/* Headers */
h1, h2, h3 {
    background: linear-gradient(135deg, #E2E8F0, #94A3B8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Selectbox / Slider labels */
label {
    color: #94A3B8 !important;
    font-weight: 500 !important;
}

/* Divider */
hr {
    border-color: rgba(124, 58, 237, 0.2) !important;
}

/* Expander */
details {
    background: rgba(30, 27, 75, 0.4) !important;
    border: 1px solid rgba(124, 58, 237, 0.2) !important;
    border-radius: 12px !important;
}

/* DataFrames */
.dataframe {
    background: rgba(15, 23, 42, 0.9) !important;
}

/* Cards via st.markdown */
.kpi-card {
    background: rgba(124, 58, 237, 0.10);
    border: 1px solid rgba(124, 58, 237, 0.30);
    border-radius: 18px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}
.kpi-card:hover {
    border-color: rgba(6, 182, 212, 0.60);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(124, 58, 237, 0.25);
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7C3AED, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.kpi-label {
    font-size: 0.85rem;
    color: #94A3B8;
    font-weight: 500;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-high   { background:#EF444420; color:#EF4444; border:1px solid #EF444450; border-radius:8px; padding:4px 12px; font-weight:600; }
.badge-medium { background:#F59E0B20; color:#F59E0B; border:1px solid #F59E0B50; border-radius:8px; padding:4px 12px; font-weight:600; }
.badge-low    { background:#10B98120; color:#10B981; border:1px solid #10B98150; border-radius:8px; padding:4px 12px; font-weight:600; }
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E2E8F0 0%, #7C3AED 50%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1.1rem;
    color: #94A3B8;
    margin-top: 0.5rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>🧠</div>
        <div style='font-size:1.1rem; font-weight:700; color:#E2E8F0;'>HR Attrition AI</div>
        <div style='font-size:0.75rem; color:#7C3AED; font-weight:500; margin-top:2px;'>
            IBM HR Analytics System
        </div>
    </div>
    <hr style='border-color:rgba(124,58,237,0.3); margin:0.5rem 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='padding:0.5rem 0;'>
        <div style='color:#94A3B8; font-size:0.75rem; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem;'>
            Navigation
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.page_link("app.py",                    label="🏠  Home",             )
    st.page_link("pages/01_Dashboard.py",     label="📊  Dashboard",        )
    st.page_link("pages/02_EDA.py",           label="🔍  Exploratory Analysis")
    st.page_link("pages/03_ML_Models.py",     label="🤖  ML Models",        )
    st.page_link("pages/04_Predict.py",       label="🔮  Predict Employee", )
    st.page_link("pages/05_Clustering.py",    label="🧩  Clustering",       )
    st.page_link("pages/06_PCA.py",           label="📉  PCA Analysis",     )

    st.markdown("<hr style='border-color:rgba(124,58,237,0.3);'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#475569; font-size:0.72rem; text-align:center; padding:0.5rem;'>
        Dataset: IBM HR Analytics<br>
        1,470 Employees · 35 Features<br>
        <span style='color:#7C3AED;'>Assignment 1 — ML Project</span>
    </div>
    """, unsafe_allow_html=True)

# ── Load data for KPIs ───────────────────────────────────────────────────────
from modules.data_loader import load_data, get_dataset_info

df   = load_data()
info = get_dataset_info(df)

# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 2rem 0 1rem 0;'>
    <div class='hero-title'>AI-Powered Employee<br>Attrition Prediction System</div>
    <div class='hero-sub'>
        Leveraging Machine Learning to identify at-risk employees, uncover attrition drivers,<br>
        and deliver actionable HR insights — powered by the <strong>IBM HR Analytics Dataset</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI Cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{info['total_employees']:,}</div>
        <div class='kpi-label'>Total Employees</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{info['attrition_rate']}%</div>
        <div class='kpi-label'>Attrition Rate</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{info['attrition_yes']}</div>
        <div class='kpi-label'>Employees Left</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>${info['avg_monthly_income']:,.0f}</div>
        <div class='kpi-label'>Avg Monthly Income</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{info['avg_age']}</div>
        <div class='kpi-label'>Avg Employee Age</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two column layout: charts + info ────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("📊 Department-wise Attrition")
    from modules import eda
    st.plotly_chart(eda.department_attrition(df), use_container_width=True)

with right:
    st.subheader("📋 System Overview")
    st.markdown("""
    <div style='background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.25);
                border-radius:16px; padding:1.5rem;'>
    <div style='color:#E2E8F0; font-weight:600; margin-bottom:1rem;'>🔬 Phase 1 — EDA</div>
    <ul style='color:#94A3B8; font-size:0.9rem; line-height:1.8; padding-left:1.2rem;'>
        <li>Attrition rate analysis</li>
        <li>Department & gender breakdown</li>
        <li>Salary & experience impact</li>
        <li>12 interactive visualizations</li>
    </ul>
    <div style='color:#E2E8F0; font-weight:600; margin-top:1rem; margin-bottom:0.5rem;'>🤖 Phase 2 — Machine Learning</div>
    <ul style='color:#94A3B8; font-size:0.9rem; line-height:1.8; padding-left:1.2rem;'>
        <li>Logistic Regression</li>
        <li>Decision Tree (HR Rules)</li>
        <li>Random Forest (Production)</li>
        <li>SVM · KNN · Naive Bayes</li>
        <li>K-Means Clustering</li>
        <li>PCA Dimensionality Reduction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Attrition by gender ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("⚡ OverTime Impact on Attrition")
    st.plotly_chart(eda.overtime_impact(df), use_container_width=True)
with col2:
    st.subheader("🎂 Age Distribution")
    st.plotly_chart(eda.age_distribution(df), use_container_width=True)

st.markdown("---")

# ── Quick Navigation Cards ───────────────────────────────────────────────────
st.subheader("🚀 Quick Navigation")
nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(6)

nav_items = [
    ("📊", "Dashboard", "Real-time KPIs & attrition overview"),
    ("🔍", "EDA", "12 interactive charts & filters"),
    ("🤖", "ML Models", "7 algorithms with metrics & ROC"),
    ("🔮", "Predict", "Live employee risk prediction"),
    ("🧩", "Clustering", "3-cluster employee segmentation"),
    ("📉", "PCA", "Dimensionality reduction analysis"),
]
for col, (icon, title, desc) in zip([nav1, nav2, nav3, nav4, nav5, nav6], nav_items):
    with col:
        st.markdown(f"""
        <div class='kpi-card' style='cursor:pointer;'>
            <div style='font-size:1.8rem;'>{icon}</div>
            <div style='color:#E2E8F0; font-weight:700; margin:0.4rem 0;'>{title}</div>
            <div style='color:#64748B; font-size:0.78rem;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#475569; font-size:0.8rem;'>
    Built with ❤️ using Streamlit · scikit-learn · Plotly &nbsp;|&nbsp;
    Dataset: IBM HR Analytics (1,470 employees, 35 features)
</div>
""", unsafe_allow_html=True)
