"""
Page 5 — K-Means Clustering
Employee segmentation into 3 clusters: High Performers, At Risk, New Employees.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Clustering | HR Attrition AI", page_icon="🧩", layout="wide")

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
div[data-testid="metric-container"] { background:rgba(124,58,237,0.12); border:1px solid rgba(124,58,237,0.3); border-radius:16px; padding:1rem; }
.cluster-card { border-radius:18px; padding:1.5rem; margin:0.5rem 0; text-align:center; }
</style>
""", unsafe_allow_html=True)

from modules.data_loader import load_data
from modules.preprocessor import preprocess
from modules.models import train_kmeans, run_pca, CLUSTER_LABELS

df = load_data()
X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders, df_enc = preprocess(df)

# Use full scaled dataset for clustering
from sklearn.preprocessing import StandardScaler
X_all = df_enc[feature_names].values
X_all_scaled = scaler.transform(X_all)

km, cluster_labels, inertias = train_kmeans(X_all_scaled, n_clusters=3)
pca_model, X_pca = run_pca(X_all_scaled, n_components=3)

# Assign cluster to original df
df_plot = df.copy()
df_plot["Cluster"] = cluster_labels
df_plot["Cluster_Label"] = df_plot["Cluster"].map({k: v[0] for k, v in CLUSTER_LABELS.items()})

# Map clusters by attrition rate (highest → At Risk)
cluster_attrition = (
    df_plot.groupby("Cluster")["Attrition"]
    .apply(lambda x: (x == "Yes").mean())
    .sort_values(ascending=False)
)
cluster_income = df_plot.groupby("Cluster")["MonthlyIncome"].mean()
cluster_years  = df_plot.groupby("Cluster")["TotalWorkingYears"].mean()

# Semantic cluster naming
cluster_order = cluster_attrition.index.tolist()
SEMANTIC = {
    cluster_order[0]: ("⚠️ At Risk",        "#EF4444"),
    cluster_order[1]: ("🏆 High Performers", "#10B981"),
    cluster_order[2]: ("🌱 New Employees",   "#06B6D4"),
}
df_plot["Cluster_Semantic"] = df_plot["Cluster"].map({k: v[0] for k, v in SEMANTIC.items()})
df_plot["Cluster_Color"]    = df_plot["Cluster"].map({k: v[1] for k, v in SEMANTIC.items()})

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2.2rem; font-weight:800;'>🧩 K-Means Employee Clustering</h1>
<p style='color:#94A3B8; margin-top:-0.5rem;'>Unsupervised segmentation of employees into 3 meaningful groups</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Cluster Summary Cards ─────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
cluster_configs = [
    (cluster_order[1], "🏆", "High Performers", "#10B981",
     "Experienced employees with high income and high satisfaction. Company backbone."),
    (cluster_order[0], "⚠️", "At Risk",         "#EF4444",
     "Low satisfaction, overtime workers, below-average income. Immediate HR attention needed."),
    (cluster_order[2], "🌱", "New Employees",   "#06B6D4",
     "Recent hires with fewer years, lower income. Growth-oriented with proper onboarding."),
]
for col, (cid, icon, label, color, desc) in zip([c1, c2, c3], cluster_configs):
    count = (df_plot["Cluster"] == cid).sum()
    att_rate = (df_plot[df_plot["Cluster"] == cid]["Attrition"] == "Yes").mean() * 100
    avg_sal  = df_plot[df_plot["Cluster"] == cid]["MonthlyIncome"].mean()
    with col:
        st.markdown(f"""
        <div class='cluster-card' style='background:rgba(30,27,75,0.7); border:2px solid {color}40;'>
            <div style='font-size:2.5rem;'>{icon}</div>
            <div style='color:{color}; font-size:1.2rem; font-weight:800; margin:0.5rem 0;'>{label}</div>
            <div style='color:#94A3B8; font-size:0.85rem; line-height:1.6; margin-bottom:1rem;'>{desc}</div>
            <div style='display:flex; justify-content:space-around;'>
                <div>
                    <div style='color:#E2E8F0; font-size:1.4rem; font-weight:700;'>{count}</div>
                    <div style='color:#64748B; font-size:0.75rem;'>Employees</div>
                </div>
                <div>
                    <div style='color:{color}; font-size:1.4rem; font-weight:700;'>{att_rate:.1f}%</div>
                    <div style='color:#64748B; font-size:0.75rem;'>Attrition</div>
                </div>
                <div>
                    <div style='color:#06B6D4; font-size:1.4rem; font-weight:700;'>${avg_sal:,.0f}</div>
                    <div style='color:#64748B; font-size:0.75rem;'>Avg Salary</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["🌐 3D Scatter", "📊 Cluster Profiles", "📈 Elbow Curve", "🗃️ Data"])

with t1:
    st.subheader("3D PCA Cluster Visualization")
    color_map = {v[0]: v[1] for v in SEMANTIC.values()}
    fig3d = go.Figure()
    for cid, (sem_label, color) in SEMANTIC.items():
        mask = df_plot["Cluster"] == cid
        fig3d.add_trace(go.Scatter3d(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            z=X_pca[mask, 2],
            mode="markers",
            name=sem_label,
            marker=dict(
                size=3.5,
                color=color,
                opacity=0.75,
                line=dict(width=0),
            ),
            hovertemplate=(
                f"<b>{sem_label}</b><br>"
                "PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>"
            ),
        ))
    fig3d.update_layout(
        template="plotly_dark",
        height=600,
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            bgcolor="#0F172A",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

with t2:
    st.subheader("Cluster Profile Comparison")
    profile_metrics = ["MonthlyIncome","JobSatisfaction","TotalWorkingYears",
                       "YearsAtCompany","WorkLifeBalance","Age"]
    profile_labels  = ["Avg Salary","Job Satisfaction","Working Years",
                       "Years at Co.","Work-Life Balance","Age"]

    cluster_profile = df_plot.groupby("Cluster_Semantic")[profile_metrics].mean().reset_index()

    fig_bar = go.Figure()
    colors_for_clusters = ["#10B981","#EF4444","#06B6D4"]
    for i, row in cluster_profile.iterrows():
        fig_bar.add_trace(go.Bar(
            name=row["Cluster_Semantic"],
            x=profile_labels,
            y=[row[m] for m in profile_metrics],
            marker_color=colors_for_clusters[i % 3],
        ))
    fig_bar.update_layout(
        barmode="group",
        template="plotly_dark",
        height=420,
        xaxis_title="Feature",
        yaxis_title="Average Value",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Attrition by cluster
    st.subheader("Attrition Rate by Cluster")
    att_by_cluster = (
        df_plot.groupby("Cluster_Semantic")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index()
    )
    att_by_cluster.columns = ["Cluster", "Attrition Rate (%)"]
    att_by_cluster = att_by_cluster.sort_values("Attrition Rate (%)", ascending=False)

    fig_att = go.Figure(go.Bar(
        x=att_by_cluster["Attrition Rate (%)"],
        y=att_by_cluster["Cluster"],
        orientation="h",
        marker=dict(
            color=["#EF4444","#F59E0B","#10B981"],
            line=dict(width=0),
        ),
        text=att_by_cluster["Attrition Rate (%)"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig_att.update_layout(
        template="plotly_dark",
        height=300,
        xaxis=dict(range=[0, 60]),
        margin=dict(l=20, r=60, t=20, b=20),
    )
    st.plotly_chart(fig_att, use_container_width=True)

with t3:
    st.subheader("📈 Elbow Curve — Optimal K Selection")
    st.markdown("""
    <div style='color:#94A3B8; font-size:0.9rem; margin-bottom:1rem;'>
    The elbow curve shows within-cluster inertia (WCSS) for different values of K.
    The optimal K is at the "elbow" where adding more clusters yields diminishing returns.
    K=3 is the clear elbow for this dataset.
    </div>""", unsafe_allow_html=True)

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(range(1, 9)),
        y=inertias,
        mode="lines+markers",
        line=dict(color="#7C3AED", width=2.5),
        marker=dict(size=8, color="#06B6D4",
                    line=dict(color="#7C3AED", width=2)),
        name="WCSS",
    ))
    fig_elbow.add_vline(x=3, line_dash="dash", line_color="#EF4444",
                        annotation_text="Optimal K=3", annotation_font_color="#EF4444")
    fig_elbow.update_layout(
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Within-Cluster Sum of Squares",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

with t4:
    st.subheader("Cluster Data Table")
    show_cols = ["Age","Department","JobRole","MonthlyIncome","JobSatisfaction",
                 "TotalWorkingYears","OverTime","Attrition","Cluster_Semantic"]
    display_df = df_plot[show_cols].copy().rename(columns={"Cluster_Semantic": "Cluster"})

    def style_cluster(val):
        if "High" in str(val):   return "color:#10B981; font-weight:700;"
        if "Risk" in str(val):   return "color:#EF4444; font-weight:700;"
        if "New" in str(val):    return "color:#06B6D4; font-weight:700;"
        return ""

    st.dataframe(
        display_df.style.applymap(style_cluster, subset=["Cluster"]),
        use_container_width=True, height=500
    )
    st.caption(f"Total: {len(display_df):,} employees across 3 clusters")
