"""
Page 3 — ML Models
All 7 models with metrics, confusion matrices, ROC curves, and feature importance.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="ML Models | HR Attrition AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#1E1B4B,#0F172A); border-right:1px solid rgba(124,58,237,0.3); }
div[data-testid="metric-container"] { background:rgba(124,58,237,0.12); border:1px solid rgba(124,58,237,0.3); border-radius:16px; padding:1rem; transition:transform 0.2s; }
div[data-testid="metric-container"]:hover { transform:translateY(-3px); }
.stTabs [data-baseweb="tab-list"] { background:rgba(15,23,42,0.8); border-radius:12px; padding:4px; gap:4px; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#7C3AED,#06B6D4) !important; color:white !important; }
h1,h2,h3 { background:linear-gradient(135deg,#E2E8F0,#94A3B8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.model-card { background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.25); border-radius:16px; padding:1.25rem; margin-bottom:1rem; }
label { color:#94A3B8 !important; font-weight:500 !important; }
</style>
""", unsafe_allow_html=True)

from modules.data_loader import load_data
from modules.preprocessor import preprocess
from modules.models import train_all_models
from modules.utils import (
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, get_decision_rules, model_accuracy_comparison
)

df = load_data()
X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders, df_enc = preprocess(df)
results = train_all_models(X_train, X_test, y_train, y_test, feature_names)

st.markdown("""
<h1 style='font-size:2.2rem; font-weight:800;'>🤖 Machine Learning Models</h1>
<p style='color:#94A3B8; margin-top:-0.5rem;'>7 algorithms trained and evaluated on IBM HR Analytics data</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Overall Comparison ────────────────────────────────────────────────────────
st.subheader("🏆 Model Accuracy Comparison")
st.plotly_chart(model_accuracy_comparison(results), use_container_width=True)

# Best model highlight
best_model = max(results, key=lambda k: results[k]["metrics"]["Accuracy"])
best_acc   = results[best_model]["metrics"]["Accuracy"]
st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(124,58,237,0.2),rgba(6,182,212,0.2));
            border:1px solid rgba(6,182,212,0.4); border-radius:16px; padding:1rem;
            text-align:center; margin-bottom:1rem;'>
    <span style='color:#94A3B8;'>🏆 Best Model: </span>
    <span style='color:#06B6D4; font-weight:700; font-size:1.1rem;'>{best_model}</span>
    <span style='color:#94A3B8;'> with </span>
    <span style='color:#10B981; font-weight:700; font-size:1.1rem;'>{best_acc:.1f}% accuracy</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Per-model Tabs ────────────────────────────────────────────────────────────
model_names = list(results.keys())
tabs = st.tabs([f"{'🏅' if n == best_model else '🔹'} {n}" for n in model_names])

for tab, name in zip(tabs, model_names):
    with tab:
        res = results[name]
        m   = res["metrics"]
        mdl = res["model"]
        y_pred = res["y_pred"]

        # Description
        st.markdown(f"""
        <div class='model-card'>
            <div style='color:#94A3B8; font-size:0.9rem; line-height:1.7;'>{res['description']}</div>
        </div>""", unsafe_allow_html=True)

        # Metrics row
        mc1, mc2, mc3, mc4 = st.columns(4)
        metric_items = [
            ("Accuracy",  m["Accuracy"],  "#06B6D4"),
            ("Precision", m["Precision"], "#7C3AED"),
            ("Recall",    m["Recall"],    "#F59E0B"),
            ("F1 Score",  m["F1 Score"],  "#10B981"),
        ]
        for col, (label, value, color) in zip([mc1, mc2, mc3, mc4], metric_items):
            with col:
                st.markdown(f"""
                <div style='background:rgba(30,27,75,0.6); border:1px solid {color}40;
                            border-radius:14px; padding:1rem; text-align:center;'>
                    <div style='font-size:1.8rem; font-weight:800; color:{color};'>{value:.1f}%</div>
                    <div style='font-size:0.8rem; color:#94A3B8; text-transform:uppercase;
                                font-weight:600; letter-spacing:0.05em;'>{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(plot_confusion_matrix(y_test, y_pred, name), use_container_width=True)
        with ch2:
            st.plotly_chart(plot_roc_curve(mdl, X_test, y_test, name), use_container_width=True)

        # Feature importance / Decision rules
        if name in ["Random Forest", "Decision Tree", "Logistic Regression"]:
            st.markdown("---")
            if name == "Decision Tree":
                fi_col, rules_col = st.columns(2)
                with fi_col:
                    st.plotly_chart(plot_feature_importance(mdl, feature_names, name),
                                    use_container_width=True)
                with rules_col:
                    st.subheader("📋 HR Decision Rules")
                    st.markdown("""
                    <div style='color:#94A3B8; font-size:0.85rem; margin-bottom:0.5rem;'>
                    Example: IF JobSatisfaction &lt; 2 AND MonthlyIncome &lt; 4000 → High Attrition Risk
                    </div>""", unsafe_allow_html=True)
                    rules = get_decision_rules(mdl, feature_names, max_depth=4)
                    st.code(rules, language="text")
            else:
                st.plotly_chart(plot_feature_importance(mdl, feature_names, name),
                                use_container_width=True)

        # KNN specific
        if name == "KNN" and "best_k" in res:
            st.info(f"🔍 Optimal k selected via grid search: **k = {res['best_k']}**")

st.markdown("---")

# ── Comparison Table ──────────────────────────────────────────────────────────
st.subheader("📊 Full Metrics Comparison Table")
import pandas as pd
rows = []
for name, res in results.items():
    m = res["metrics"]
    rows.append({
        "Model": name,
        "Accuracy (%)": m["Accuracy"],
        "Precision (%)": m["Precision"],
        "Recall (%)": m["Recall"],
        "F1 Score (%)": m["F1 Score"],
    })

comp_df = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False)

def highlight_best(s):
    is_max = s == s.max()
    return ['background: rgba(6,182,212,0.2); color:#06B6D4; font-weight:700' if v else '' for v in is_max]

styled = comp_df.style.apply(highlight_best, subset=["Accuracy (%)","Precision (%)","Recall (%)","F1 Score (%)"])
st.dataframe(styled, use_container_width=True, hide_index=True)
