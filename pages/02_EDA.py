"""
Page 2 — Exploratory Data Analysis
Interactive filters and 12+ Plotly charts.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="EDA | HR Attrition AI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#1E1B4B,#0F172A); border-right:1px solid rgba(124,58,237,0.3); }
div[data-testid="metric-container"] { background:rgba(124,58,237,0.12); border:1px solid rgba(124,58,237,0.3); border-radius:16px; padding:1rem; transition:transform 0.2s; }
div[data-testid="metric-container"]:hover { transform:translateY(-3px); }
.stTabs [data-baseweb="tab-list"] { background:rgba(15,23,42,0.8); border-radius:12px; padding:4px; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#7C3AED,#06B6D4) !important; color:white !important; }
h1,h2,h3 { background:linear-gradient(135deg,#E2E8F0,#94A3B8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
label { color:#94A3B8 !important; font-weight:500 !important; }
</style>
""", unsafe_allow_html=True)

from modules.data_loader import load_data
from modules import eda

df = load_data()

st.markdown("""
<h1 style='font-size:2.2rem; font-weight:800;'>🔍 Exploratory Data Analysis</h1>
<p style='color:#94A3B8; margin-top:-0.5rem;'>Uncover patterns and drivers behind employee attrition</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Filters ───────────────────────────────────────────────────────────────────
with st.expander("🎛️ Filters — Click to expand", expanded=True):
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        depts = ["All"] + sorted(df["Department"].unique().tolist())
        sel_dept = st.selectbox("Department", depts, key="eda_dept")
    with f2:
        genders = ["All"] + sorted(df["Gender"].unique().tolist())
        sel_gender = st.selectbox("Gender", genders, key="eda_gender")
    with f3:
        roles = ["All"] + sorted(df["JobRole"].unique().tolist())
        sel_role = st.selectbox("Job Role", roles, key="eda_role")
    with f4:
        age_range = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()),
                              (18, 60), key="eda_age")

# Apply filters
fdf = df.copy()
if sel_dept   != "All": fdf = fdf[fdf["Department"] == sel_dept]
if sel_gender != "All": fdf = fdf[fdf["Gender"] == sel_gender]
if sel_role   != "All": fdf = fdf[fdf["JobRole"] == sel_role]
fdf = fdf[(fdf["Age"] >= age_range[0]) & (fdf["Age"] <= age_range[1])]

# Filtered KPIs
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Filtered Employees", f"{len(fdf):,}", f"{len(fdf)-len(df):+,}")
m2.metric("Attrition Count",
          (fdf["Attrition"] == "Yes").sum(),
          f"{(fdf['Attrition']=='Yes').sum()-(df['Attrition']=='Yes').sum():+,}")
m3.metric("Attrition Rate",
          f"{(fdf['Attrition']=='Yes').mean()*100:.1f}%")
m4.metric("Avg Monthly Income",
          f"${fdf['MonthlyIncome'].mean():,.0f}")

st.markdown("---")

# ── Charts ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🏢 Department", "💼 Job & Role",
    "💰 Compensation", "😊 Satisfaction", "🔗 Correlations"
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🥧 Attrition Rate")
        st.plotly_chart(eda.attrition_rate_donut(fdf), use_container_width=True)
    with c2:
        st.subheader("🎂 Age Distribution")
        st.plotly_chart(eda.age_distribution(fdf), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("💍 Marital Status vs Attrition")
        st.plotly_chart(eda.marital_status_attrition(fdf), use_container_width=True)
    with c4:
        st.subheader("⚡ OverTime Impact")
        st.plotly_chart(eda.overtime_impact(fdf), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏢 Department-wise Attrition")
        st.plotly_chart(eda.department_attrition(fdf), use_container_width=True)
    with c2:
        st.subheader("🚻 Gender-wise Attrition Rate")
        st.plotly_chart(eda.gender_attrition(fdf), use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💼 Job Role Attrition Rate")
        st.plotly_chart(eda.job_role_attrition(fdf), use_container_width=True)
    with c2:
        st.subheader("📅 Years at Company")
        st.plotly_chart(eda.years_at_company_attrition(fdf), use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💰 Monthly Income vs Attrition")
        st.plotly_chart(eda.salary_vs_attrition(fdf), use_container_width=True)
    with c2:
        st.subheader("📊 Total Working Years vs Attrition")
        st.plotly_chart(eda.experience_vs_attrition(fdf), use_container_width=True)

with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("😊 Job Satisfaction vs Attrition")
        st.plotly_chart(eda.job_satisfaction_attrition(fdf), use_container_width=True)
    with c2:
        st.subheader("⚡ OverTime vs Attrition")
        st.plotly_chart(eda.overtime_impact(fdf), use_container_width=True)

with tab6:
    st.subheader("🔗 Feature Correlation Heatmap")
    st.markdown("""
    <div style='color:#94A3B8; font-size:0.9rem; margin-bottom:1rem;'>
    Shows the top 15 features most correlated with Attrition. 
    Positive values → increases attrition risk. Negative values → decreases risk.
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(eda.correlation_heatmap(fdf), use_container_width=True)

st.markdown("---")

# ── Raw Data Explorer ─────────────────────────────────────────────────────────
with st.expander("🗃️ Raw Data Explorer"):
    st.markdown(f"Showing **{len(fdf):,}** rows after filters")
    st.dataframe(
        fdf.head(200).style.applymap(
            lambda v: "color: #EF4444; font-weight:600;" if v == "Yes" else
                      "color: #10B981;" if v == "No" else "",
            subset=["Attrition"]
        ),
        use_container_width=True,
    )
    st.caption("Showing first 200 rows. Apply filters to narrow down.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset Shape:**")
        st.code(f"Rows: {fdf.shape[0]}, Columns: {fdf.shape[1]}")
    with col2:
        st.markdown("**Missing Values:**")
        nulls = fdf.isnull().sum()
        st.code(f"Total missing: {nulls.sum()} (Dataset is clean ✅)")
