"""
Page 6 — PCA Analysis
Dimensionality reduction with explained variance, 2D/3D biplots, and feature loadings.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PCA | HR Attrition AI", page_icon="📉", layout="wide")

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
</style>
""", unsafe_allow_html=True)

from modules.data_loader import load_data
from modules.preprocessor import preprocess
from modules.models import run_pca

df = load_data()
X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders, df_enc = preprocess(df)

X_all = df_enc[feature_names].values
X_all_scaled = scaler.transform(X_all)
y_all = df_enc["Attrition"].values

# Run full PCA
pca_full, X_pca_full = run_pca(X_all_scaled, n_components=None)

ev_ratio = pca_full.explained_variance_ratio_
cum_var  = np.cumsum(ev_ratio)
n_95     = int(np.argmax(cum_var >= 0.95)) + 1
n_90     = int(np.argmax(cum_var >= 0.90)) + 1

st.markdown("""
<h1 style='font-size:2.2rem; font-weight:800;'>📉 PCA — Dimensionality Reduction</h1>
<p style='color:#94A3B8; margin-top:-0.5rem;'>Reducing 30+ features to principal components while preserving variance</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Original Features", len(feature_names))
k2.metric("Components for 90% Variance", n_90)
k3.metric("Components for 95% Variance", n_95)
k4.metric("Variance by PC1+PC2", f"{(ev_ratio[0]+ev_ratio[1])*100:.1f}%")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Explained Variance", "🔵 2D Biplot", "🌐 3D Scatter", "🔥 Feature Loadings"
])

with tab1:
    st.subheader("Explained Variance Ratio per Component")
    st.markdown("""
    <div style='color:#94A3B8; font-size:0.9rem; margin-bottom:1rem;'>
    Shows how much variance each principal component captures.
    The cumulative curve shows how many components are needed to explain a target % of data variance.
    </div>""", unsafe_allow_html=True)

    n_show = min(20, len(ev_ratio))
    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(n_show)],
        y=ev_ratio[:n_show] * 100,
        name="Individual Variance",
        marker=dict(
            color=ev_ratio[:n_show] * 100,
            colorscale=[[0, "#7C3AED"], [1, "#06B6D4"]],
            showscale=False,
        ),
    ))
    fig_var.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(n_show)],
        y=cum_var[:n_show] * 100,
        mode="lines+markers",
        name="Cumulative Variance",
        line=dict(color="#F59E0B", width=2.5),
        marker=dict(size=6),
        yaxis="y2",
    ))
    fig_var.add_hline(y=95, line_dash="dash", line_color="#EF4444",
                      yref="y2",
                      annotation_text="95% threshold",
                      annotation_font_color="#EF4444")
    fig_var.add_hline(y=90, line_dash="dot", line_color="#10B981",
                      yref="y2",
                      annotation_text="90% threshold",
                      annotation_font_color="#10B981")
    fig_var.update_layout(
        template="plotly_dark",
        height=420,
        yaxis=dict(title="Individual Variance (%)"),
        yaxis2=dict(title="Cumulative Variance (%)", overlaying="y", side="right",
                    range=[0, 105]),
        margin=dict(l=20, r=60, t=20, b=20),
        legend=dict(x=0.4, y=0.95),
    )
    st.plotly_chart(fig_var, use_container_width=True)

    st.markdown(f"""
    <div style='background:rgba(124,58,237,0.1); border:1px solid rgba(124,58,237,0.3);
                border-radius:14px; padding:1rem; margin-top:0.5rem;'>
        <span style='color:#E2E8F0; font-weight:600;'>Key Insight: </span>
        <span style='color:#94A3B8;'>
        You need <strong style='color:#06B6D4;'>{n_90} components</strong> to explain 90% of variance
        and <strong style='color:#F59E0B;'>{n_95} components</strong> to explain 95% of variance,
        reducing dimensionality from <strong style='color:#7C3AED;'>{len(feature_names)} features</strong>.
        </span>
    </div>""", unsafe_allow_html=True)

with tab2:
    st.subheader("2D PCA Biplot — Attrition Colour-Coded")
    st.markdown("""
    <div style='color:#94A3B8; font-size:0.9rem; margin-bottom:1rem;'>
    Each point is an employee projected onto the first two principal components.
    Red = attrition, Green = stayed. Separation indicates how well PCA distinguishes the classes.
    </div>""", unsafe_allow_html=True)

    pc1 = X_pca_full[:, 0]
    pc2 = X_pca_full[:, 1]

    fig2d = go.Figure()
    for label, color, marker in [(0, "#10B981", "circle"), (1, "#EF4444", "x")]:
        mask = y_all == label
        fig2d.add_trace(go.Scatter(
            x=pc1[mask], y=pc2[mask],
            mode="markers",
            name=["No Attrition", "Attrition"][label],
            marker=dict(
                size=4,
                color=color,
                symbol=marker,
                opacity=0.6,
                line=dict(width=0),
            ),
        ))

    # Feature loading arrows (top 8 by PC1+PC2 magnitude)
    loadings = pca_full.components_[:2].T  # shape (n_features, 2)
    loading_magnitude = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_idx = np.argsort(loading_magnitude)[-8:]
    scale = 4.0

    for i in top_idx:
        lx, ly = loadings[i, 0] * scale, loadings[i, 1] * scale
        fig2d.add_annotation(
            ax=0, ay=0, x=lx, y=ly,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1, arrowwidth=1.5,
            arrowcolor="#F59E0B",
        )
        fig2d.add_annotation(
            x=lx * 1.1, y=ly * 1.1,
            text=feature_names[i],
            font=dict(size=9, color="#F59E0B"),
            showarrow=False,
        )

    fig2d.update_layout(
        xaxis_title=f"PC1 ({ev_ratio[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({ev_ratio[1]*100:.1f}% variance)",
        template="plotly_dark",
        height=550,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig2d, use_container_width=True)

with tab3:
    st.subheader("3D PCA Scatter — PC1, PC2, PC3")
    pc3 = X_pca_full[:, 2]
    fig3d = go.Figure()
    for label, color, sym in [(0, "#10B981", "circle"), (1, "#EF4444", "x")]:
        mask = y_all == label
        fig3d.add_trace(go.Scatter3d(
            x=pc1[mask], y=pc2[mask], z=pc3[mask],
            mode="markers",
            name=["No Attrition", "Attrition"][label],
            marker=dict(size=3, color=color, opacity=0.6, symbol=sym),
        ))
    fig3d.update_layout(
        template="plotly_dark",
        height=580,
        scene=dict(
            xaxis_title=f"PC1 ({ev_ratio[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({ev_ratio[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({ev_ratio[2]*100:.1f}%)",
            bgcolor="#0F172A",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

with tab4:
    st.subheader("Feature Loading Heatmap")
    st.markdown("""
    <div style='color:#94A3B8; font-size:0.9rem; margin-bottom:1rem;'>
    Shows the contribution (loading) of each original feature to the first 8 principal components.
    High positive/negative values indicate strong influence on that component.
    </div>""", unsafe_allow_html=True)

    n_pc_show = min(8, len(ev_ratio))
    loadings_df = pd.DataFrame(
        pca_full.components_[:n_pc_show].T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_pc_show)],
    )
    # Sort by magnitude of first PC
    loadings_df = loadings_df.reindex(
        loadings_df["PC1"].abs().sort_values(ascending=False).index
    )

    fig_heat = go.Figure(go.Heatmap(
        z=loadings_df.values,
        x=loadings_df.columns.tolist(),
        y=loadings_df.index.tolist(),
        colorscale=[[0, "#EF4444"], [0.5, "#1E293B"], [1, "#06B6D4"]],
        zmin=-1, zmax=1,
        text=np.round(loadings_df.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        showscale=True,
    ))
    fig_heat.update_layout(
        template="plotly_dark",
        height=680,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
