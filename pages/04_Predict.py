"""
Page 4 — Predict Employee Attrition
Interactive live prediction with all models + risk badge + similar employees via KNN.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Predict | HR Attrition AI", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#1E1B4B,#0F172A); border-right:1px solid rgba(124,58,237,0.3); }
.stTabs [data-baseweb="tab-list"] { background:rgba(15,23,42,0.8); border-radius:12px; padding:4px; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#7C3AED,#06B6D4) !important; color:white !important; }
h1,h2,h3 { background:linear-gradient(135deg,#E2E8F0,#94A3B8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
label { color:#94A3B8 !important; font-weight:500 !important; }
.stButton > button { background:linear-gradient(135deg,#7C3AED,#06B6D4); color:white; border:none; border-radius:12px; font-weight:700; font-size:1.05rem; padding:0.75rem 2rem; box-shadow:0 4px 15px rgba(124,58,237,0.4); transition:all 0.3s; }
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 25px rgba(124,58,237,0.6); }
.risk-badge { border-radius:12px; padding:1rem 1.5rem; text-align:center; font-size:1.5rem; font-weight:800; margin:0.5rem 0; }
.model-result { background:rgba(30,27,75,0.5); border:1px solid rgba(124,58,237,0.25); border-radius:12px; padding:1rem; margin:0.4rem 0; }
div[data-testid="metric-container"] { background:rgba(124,58,237,0.12); border:1px solid rgba(124,58,237,0.3); border-radius:16px; padding:1rem; }
</style>
""", unsafe_allow_html=True)

from modules.data_loader import load_data
from modules.preprocessor import preprocess, encode_single_employee
from modules.models import train_all_models
from modules.utils import risk_category, get_decision_rules

df = load_data()
X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders, df_enc = preprocess(df)
results = train_all_models(X_train, X_test, y_train, y_test, feature_names)

st.markdown("""
<h1 style='font-size:2.2rem; font-weight:800;'>🔮 Predict Employee Attrition</h1>
<p style='color:#94A3B8; margin-top:-0.5rem;'>Enter employee details to predict attrition risk across all models</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("👤 Employee Profile")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📋 Personal Info**")
        age           = st.slider("Age", 18, 60, 35, key="pred_age")
        gender        = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
        marital       = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="pred_marital")
        education     = st.selectbox("Education Level",
                                     [1, 2, 3, 4, 5],
                                     format_func=lambda x: {1:"Below College",2:"College",
                                                             3:"Bachelor",4:"Master",5:"Doctor"}[x],
                                     key="pred_edu")
        edu_field     = st.selectbox("Education Field",
                                     ["Life Sciences","Medical","Marketing",
                                      "Technical Degree","Other","Human Resources"],
                                     key="pred_edu_field")
        dist_home     = st.slider("Distance from Home (km)", 1, 30, 10, key="pred_dist")

    with col2:
        st.markdown("**🏢 Job Info**")
        department    = st.selectbox("Department",
                                     ["Sales","Research & Development","Human Resources"],
                                     key="pred_dept")
        job_role      = st.selectbox("Job Role",
                                     ["Sales Executive","Research Scientist","Laboratory Technician",
                                      "Manufacturing Director","Healthcare Representative",
                                      "Manager","Sales Representative","Research Director",
                                      "Human Resources"],
                                     key="pred_role")
        job_level     = st.slider("Job Level", 1, 5, 2, key="pred_joblevel")
        job_sat       = st.selectbox("Job Satisfaction",
                                     [1, 2, 3, 4],
                                     format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x],
                                     key="pred_jobsat")
        env_sat       = st.selectbox("Environment Satisfaction",
                                     [1, 2, 3, 4],
                                     format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x],
                                     key="pred_envsat")
        overtime      = st.selectbox("OverTime", ["Yes", "No"], key="pred_ot")
        business_travel = st.selectbox("Business Travel",
                                       ["Travel_Rarely","Travel_Frequently","Non-Travel"],
                                       key="pred_bt")

    with col3:
        st.markdown("**💰 Compensation & Experience**")
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, 500, key="pred_income")
        pct_hike      = st.slider("Salary Hike (%)", 11, 25, 15, key="pred_hike")
        stock_level   = st.slider("Stock Option Level", 0, 3, 1, key="pred_stock")
        total_years   = st.slider("Total Working Years", 0, 40, 8, key="pred_total_years")
        years_company = st.slider("Years at Company", 0, 40, 4, key="pred_years_co")
        years_role    = st.slider("Years in Current Role", 0, 18, 3, key="pred_years_role")
        years_promo   = st.slider("Years Since Last Promotion", 0, 15, 2, key="pred_years_promo")
        years_mgr     = st.slider("Years with Current Manager", 0, 17, 3, key="pred_years_mgr")
        num_companies = st.slider("Num Companies Worked", 0, 9, 2, key="pred_numco")
        training      = st.slider("Trainings Last Year", 0, 6, 3, key="pred_training")
        wlb           = st.selectbox("Work-Life Balance",
                                     [1, 2, 3, 4],
                                     format_func=lambda x: {1:"Bad",2:"Good",3:"Better",4:"Best"}[x],
                                     key="pred_wlb")
        rel_sat       = st.selectbox("Relationship Satisfaction",
                                     [1, 2, 3, 4],
                                     format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x],
                                     key="pred_relsat")
        job_inv       = st.selectbox("Job Involvement",
                                     [1, 2, 3, 4],
                                     format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x],
                                     key="pred_jobinv")
        perf_rating   = st.selectbox("Performance Rating",
                                     [3, 4],
                                     format_func=lambda x: {3:"Excellent",4:"Outstanding"}[x],
                                     key="pred_perf")
        daily_rate    = st.number_input("Daily Rate", 100, 1500, 800, key="pred_drate")
        hourly_rate   = st.number_input("Hourly Rate", 30, 100, 65, key="pred_hrate")
        monthly_rate  = st.number_input("Monthly Rate", 2000, 27000, 14000, key="pred_mrate")

    submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

# ── Prediction Results ────────────────────────────────────────────────────────
if submitted:
    employee = {
        "Age": age, "BusinessTravel": business_travel, "DailyRate": daily_rate,
        "Department": department, "DistanceFromHome": dist_home, "Education": education,
        "EducationField": edu_field, "EnvironmentSatisfaction": env_sat, "Gender": gender,
        "HourlyRate": hourly_rate, "JobInvolvement": job_inv, "JobLevel": job_level,
        "JobRole": job_role, "JobSatisfaction": job_sat, "MaritalStatus": marital,
        "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies, "OverTime": overtime,
        "PercentSalaryHike": pct_hike, "PerformanceRating": perf_rating,
        "RelationshipSatisfaction": rel_sat, "StockOptionLevel": stock_level,
        "TotalWorkingYears": total_years, "TrainingTimesLastYear": training,
        "WorkLifeBalance": wlb, "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role, "YearsSinceLastPromotion": years_promo,
        "YearsWithCurrManager": years_mgr,
    }

    X_emp = encode_single_employee(employee, feature_names, label_encoders, scaler)

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    # Collect predictions from all models
    all_probs = []
    model_preds = {}
    for name, res in results.items():
        mdl = res["model"]
        pred = int(mdl.predict(X_emp)[0])
        if hasattr(mdl, "predict_proba"):
            prob = float(mdl.predict_proba(X_emp)[0][1])
        elif hasattr(mdl, "decision_function"):
            raw = mdl.decision_function(X_emp)[0]
            prob = float(1 / (1 + np.exp(-raw)))
        else:
            prob = float(pred)
        all_probs.append(prob)
        model_preds[name] = {"pred": pred, "prob": prob}

    ensemble_prob = float(np.mean(all_probs))
    risk_label, risk_color = risk_category(ensemble_prob)

    # ── Main Risk Badge ───────────────────────────────────────────────────────
    r1, r2, r3 = st.columns([1, 2, 1])
    with r2:
        attrition_verdict = "LIKELY TO LEAVE" if ensemble_prob >= 0.5 else "LIKELY TO STAY"
        verdict_color = "#EF4444" if ensemble_prob >= 0.5 else "#10B981"
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(124,58,237,0.2),rgba(6,182,212,0.15));
                    border:2px solid {risk_color}60; border-radius:20px; padding:2rem;
                    text-align:center; margin:1rem 0;'>
            <div style='font-size:3rem; margin-bottom:0.5rem;'>
                {"🚨" if ensemble_prob >= 0.65 else "⚠️" if ensemble_prob >= 0.35 else "✅"}
            </div>
            <div style='font-size:1.8rem; font-weight:800; color:{verdict_color};'>
                {attrition_verdict}
            </div>
            <div style='font-size:1.1rem; color:#94A3B8; margin:0.5rem 0;'>
                Ensemble Attrition Probability
            </div>
            <div style='font-size:2.5rem; font-weight:900; color:{risk_color};'>
                {ensemble_prob*100:.1f}%
            </div>
            <div style='font-size:1.1rem; font-weight:700; color:{risk_color}; margin-top:0.5rem;'>
                {risk_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-model breakdown ───────────────────────────────────────────────────
    st.subheader("🤖 Individual Model Predictions")
    m_cols = st.columns(3)
    for i, (name, mp) in enumerate(model_preds.items()):
        col = m_cols[i % 3]
        p    = mp["prob"]
        pred = mp["pred"]
        bar_color = "#EF4444" if pred == 1 else "#10B981"
        label = "⚠️ Attrition" if pred == 1 else "✅ Stays"
        with col:
            st.markdown(f"""
            <div class='model-result'>
                <div style='color:#E2E8F0; font-weight:600; font-size:0.9rem;'>{name}</div>
                <div style='color:{bar_color}; font-weight:700; font-size:1.2rem;'>{label}</div>
                <div style='background:#1E293B; border-radius:8px; height:8px; margin:0.5rem 0;'>
                    <div style='background:{bar_color}; border-radius:8px; height:8px;
                                width:{p*100:.0f}%;'></div>
                </div>
                <div style='color:#94A3B8; font-size:0.8rem;'>Probability: {p*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Probability Radar Chart ───────────────────────────────────────────────
    st.subheader("📡 Model Probability Comparison")
    r_names = list(model_preds.keys())
    r_probs = [model_preds[n]["prob"] * 100 for n in r_names]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=r_probs + [r_probs[0]],
        theta=r_names + [r_names[0]],
        fill="toself",
        fillcolor="rgba(124,58,237,0.2)",
        line=dict(color="#7C3AED", width=2),
        name="Attrition Probability",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[50]*len(r_names) + [50],
        theta=r_names + [r_names[0]],
        line=dict(color="#EF4444", dash="dash", width=1),
        name="Decision Threshold (50%)",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(15,23,42,0.8)",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#94A3B8")),
            angularaxis=dict(tickfont=dict(color="#E2E8F0")),
        ),
        showlegend=True,
        template="plotly_dark",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # ── Key Risk Factors ──────────────────────────────────────────────────────
    st.subheader("⚠️ Key Risk Factors for This Employee")
    risk_factors = []
    if job_sat <= 2:
        risk_factors.append(("🔴 Low Job Satisfaction", f"Score: {job_sat}/4 — Strong attrition predictor"))
    if overtime == "Yes":
        risk_factors.append(("🔴 Working OverTime", "Employees with overtime are 3× more likely to leave"))
    if monthly_income < 3500:
        risk_factors.append(("🔴 Below-Average Income", f"${monthly_income:,}/mo is in the high-risk range"))
    if years_promo > 5:
        risk_factors.append(("🟡 No Recent Promotion", f"{years_promo} years since last promotion"))
    if dist_home > 20:
        risk_factors.append(("🟡 Long Commute", f"{dist_home} km from home"))
    if marital == "Single":
        risk_factors.append(("🟡 Marital Status: Single", "Single employees tend to have higher mobility"))
    if business_travel == "Travel_Frequently":
        risk_factors.append(("🟡 Frequent Business Travel", "High travel frequency increases burnout risk"))
    if wlb <= 2:
        risk_factors.append(("🟡 Poor Work-Life Balance", f"Score: {wlb}/4"))
    if total_years < 3:
        risk_factors.append(("🟢 Early Career", "New employees have higher initial attrition rate"))
    if not risk_factors:
        risk_factors.append(("🟢 No Major Risk Factors", "This employee's profile appears stable"))

    for emoji_label, desc in risk_factors:
        color = "#EF4444" if "🔴" in emoji_label else "#F59E0B" if "🟡" in emoji_label else "#10B981"
        st.markdown(f"""
        <div style='background:rgba(30,27,75,0.5); border-left:4px solid {color};
                    border-radius:0 12px 12px 0; padding:0.75rem 1rem; margin:0.4rem 0;'>
            <span style='color:{color}; font-weight:700;'>{emoji_label}</span>
            <span style='color:#94A3B8; font-size:0.9rem; margin-left:0.5rem;'>— {desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Decision Tree Rules ────────────────────────────────────────────────────
    st.subheader("📋 HR Decision Rules (Decision Tree)")
    with st.expander("View Full Decision Tree Rules"):
        dt_model = results["Decision Tree"]["model"]
        rules = get_decision_rules(dt_model, feature_names, max_depth=4)
        st.code(rules, language="text")

    # ── Similar Employees (KNN) ────────────────────────────────────────────────
    st.subheader("👥 Similar Employees (KNN Neighbours)")
    knn_model = results["KNN"]["model"]
    distances, indices = knn_model.kneighbors(X_emp, n_neighbors=5)
    similar = df.iloc[indices[0]][["Age","Department","JobRole","MonthlyIncome",
                                   "JobSatisfaction","TotalWorkingYears","Attrition"]].copy()
    similar["Similarity"] = [f"{(1/(1+d))*100:.1f}%" for d in distances[0]]
    similar = similar.reset_index(drop=True)
    st.markdown("""
    <div style='color:#94A3B8; font-size:0.85rem; margin-bottom:0.5rem;'>
    These are the 5 most similar employees in the dataset (found by KNN). 
    Their attrition status gives additional context.
    </div>""", unsafe_allow_html=True)
    st.dataframe(similar, use_container_width=True, hide_index=True)
