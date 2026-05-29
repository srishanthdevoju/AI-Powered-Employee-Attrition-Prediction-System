"""
Page 1 — Dashboard
Real-time KPIs, attrition gauges, department breakdown, and risk summary.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Dashboard | HR Attrition AI", page_icon="📊", layout="wide")

# Reuse global CSS via a helper
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1E1B4B 0%, #0F172A 100%); border-right: 1px solid rgba(124,58,237,0.3); }
div[data-testid="metric-container"] { background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.3); border-radius: 16px; padding: 1rem; backdrop-filter: blur(10px); transition: transform 0.2s; }
div[data-testid="metric-container"]:hover { transform: translateY(-3px); border-color: rgba(6,182,212,0.6); }
.stTabs [data-baseweb="tab-list"] { background: rgba(15,23,42,0.8); border-radius: 12px; padding: 4px; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #7C3AED, #06B6D4) !important; color: white !important; }
.stButton > button { background: linear-gradient(135deg,#7C3AED,#06B6D4); color:white; border:none; border-radius:12px; font-weight:600; }
.kpi-card { background: rgba(124,58,237,0.10); border: 1px solid rgba(124,58,237,0.30); border-radius: 18px; padding: 1.5rem; text-align: center; backdrop-filter: blur(12px); transition: all 0.3s; }
.kpi-card:hover { border-color: rgba(6,182,212,0.60); transform: translateY(-4px); box-shadow: 0 12px 40px rgba(124,58,237,0.25); }
.kpi-value { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg,#7C3AED,#06B6D4); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.kpi-label { font-size: 0.85rem; color: #94A3B8; font-weight: 500; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
h1,h2,h3 { background: linear-gradient(135deg,#E2E8F0,#94A3B8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
</style>
""", unsafe_allow_html=True)

from modules.data_loader import load_data, get_dataset_info
from modules import eda

df   = load_data()
info = get_dataset_info(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2.2rem; font-weight:800;'>📊 HR Attrition Dashboard</h1>
<p style='color:#94A3B8; margin-top:-0.5rem;'>Real-time workforce analytics & attrition intelligence</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI Row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    (info['total_employees'], "Total Employees", "👥"),
    (f"{info['attrition_rate']}%", "Attrition Rate", "📉"),
    (info['attrition_yes'], "Employees Left", "🚪"),
    (info['attrition_no'], "Retained", "✅"),
    (f"${info['avg_monthly_income']:,.0f}", "Avg Salary/Mo", "💰"),
    (f"{info['avg_age']} yrs", "Avg Age", "🎂"),
]
for col, (val, lbl, icon) in zip([c1,c2,c3,c4,c5,c6], kpis):
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size:1.4rem;'>{icon}</div>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Gauges + donut ────────────────────────────────────────────────────
g1, g2, g3 = st.columns([2, 2, 3])

with g1:
    # Attrition Rate Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=info['attrition_rate'],
        delta={"reference": 10, "suffix": "%"},
        title={"text": "Attrition Rate", "font": {"color": "#94A3B8", "size": 14}},
        number={"suffix": "%", "font": {"color": "#E2E8F0", "size": 32}},
        gauge={
            "axis": {"range": [0, 40], "tickcolor": "#475569"},
            "bar": {"color": "#7C3AED"},
            "bgcolor": "#1E293B",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 10],  "color": "#10B98120"},
                {"range": [10, 20], "color": "#F59E0B20"},
                {"range": [20, 40], "color": "#EF444420"},
            ],
            "threshold": {
                "line": {"color": "#EF4444", "width": 3},
                "thickness": 0.75,
                "value": 20,
            },
        },
    ))
    fig_gauge.update_layout(
        template="plotly_dark", height=270,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with g2:
    # Avg Salary Gauge
    avg_sal = info['avg_monthly_income']
    fig_sal = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_sal,
        title={"text": "Avg Monthly Income ($)", "font": {"color": "#94A3B8", "size": 14}},
        number={"prefix": "$", "font": {"color": "#E2E8F0", "size": 28}},
        gauge={
            "axis": {"range": [0, 20000], "tickcolor": "#475569"},
            "bar": {"color": "#06B6D4"},
            "bgcolor": "#1E293B",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 5000],   "color": "#EF444420"},
                {"range": [5000, 12000],"color": "#F59E0B20"},
                {"range": [12000, 20000],"color": "#10B98120"},
            ],
        },
    ))
    fig_sal.update_layout(
        template="plotly_dark", height=270,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_sal, use_container_width=True)

with g3:
    st.plotly_chart(eda.attrition_rate_donut(df), use_container_width=True)

st.markdown("---")

# ── Row 3: Department & Job Role ─────────────────────────────────────────────
r1, r2 = st.columns(2)
with r1:
    st.subheader("🏢 Department Attrition")
    st.plotly_chart(eda.department_attrition(df), use_container_width=True)
with r2:
    st.subheader("💼 Job Role Attrition Rate")
    st.plotly_chart(eda.job_role_attrition(df), use_container_width=True)

st.markdown("---")

# ── Row 4: Salary + Experience ───────────────────────────────────────────────
r3, r4 = st.columns(2)
with r3:
    st.subheader("💰 Salary Impact")
    st.plotly_chart(eda.salary_vs_attrition(df), use_container_width=True)
with r4:
    st.subheader("📅 Experience Impact")
    st.plotly_chart(eda.experience_vs_attrition(df), use_container_width=True)

st.markdown("---")

# ── Dept Table ───────────────────────────────────────────────────────────────
st.subheader("📋 Department Summary Table")
dept_summary = (
    df.groupby("Department")
    .agg(
        Total=("Attrition", "count"),
        Attrited=("Attrition", lambda x: (x == "Yes").sum()),
        Avg_Salary=("MonthlyIncome", "mean"),
        Avg_Age=("Age", "mean"),
    )
    .reset_index()
)
dept_summary["Attrition_Rate"] = (dept_summary["Attrited"] / dept_summary["Total"] * 100).round(1).astype(str) + "%"
dept_summary["Avg_Salary"] = dept_summary["Avg_Salary"].round(0).astype(int).apply(lambda x: f"${x:,}")
dept_summary["Avg_Age"] = dept_summary["Avg_Age"].round(1)
dept_summary = dept_summary.rename(columns={
    "Department": "Department", "Total": "Total Employees",
    "Attrited": "Left", "Avg_Salary": "Avg Salary",
    "Avg_Age": "Avg Age", "Attrition_Rate": "Attrition Rate"
})
st.dataframe(dept_summary, use_container_width=True, hide_index=True)
