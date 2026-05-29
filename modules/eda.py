"""
EDA Module
All Exploratory Data Analysis charts using Plotly with dark theme.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_dark"
PRIMARY  = "#7C3AED"
ACCENT   = "#06B6D4"
DANGER   = "#EF4444"
SUCCESS  = "#10B981"
WARNING  = "#F59E0B"

COLOR_MAP = {"Yes": DANGER, "No": SUCCESS}


def attrition_rate_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["Attrition"].value_counts().reset_index()
    counts.columns = ["Attrition", "Count"]
    fig = go.Figure(go.Pie(
        labels=counts["Attrition"],
        values=counts["Count"],
        hole=0.65,
        marker=dict(colors=[SUCCESS, DANGER]),
        textinfo="label+percent",
        textfont=dict(size=14),
    ))
    fig.update_layout(
        title="Overall Attrition Rate",
        template=TEMPLATE,
        showlegend=True,
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        annotations=[dict(
            text=f"{(df['Attrition']=='Yes').mean()*100:.1f}%<br>Attrition",
            x=0.5, y=0.5, font_size=16, showarrow=False,
            font_color="white"
        )],
    )
    return fig


def department_attrition(df: pd.DataFrame) -> go.Figure:
    grp = (
        df.groupby(["Department", "Attrition"])
        .size().reset_index(name="Count")
    )
    fig = px.bar(
        grp, x="Department", y="Count", color="Attrition",
        barmode="group",
        color_discrete_map=COLOR_MAP,
        title="Department-wise Attrition",
        template=TEMPLATE,
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def gender_attrition(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["Gender", "Attrition"]).size().reset_index(name="Count")
    total = df.groupby("Gender").size().reset_index(name="Total")
    grp = grp.merge(total, on="Gender")
    grp["Percentage"] = grp["Count"] / grp["Total"] * 100

    fig = px.bar(
        grp[grp["Attrition"] == "Yes"],
        x="Gender", y="Percentage",
        color="Gender",
        color_discrete_map={"Male": ACCENT, "Female": PRIMARY},
        title="Gender-wise Attrition Rate (%)",
        template=TEMPLATE,
        text=grp[grp["Attrition"] == "Yes"]["Percentage"].round(1).astype(str) + "%",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def salary_vs_attrition(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="Attrition", y="MonthlyIncome",
        color="Attrition",
        color_discrete_map=COLOR_MAP,
        title="Monthly Income vs Attrition",
        template=TEMPLATE,
        points="outliers",
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def experience_vs_attrition(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df, x="Attrition", y="TotalWorkingYears",
        color="Attrition",
        color_discrete_map=COLOR_MAP,
        box=True,
        title="Total Working Years vs Attrition",
        template=TEMPLATE,
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def overtime_impact(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
    total = df.groupby("OverTime").size().reset_index(name="Total")
    grp = grp.merge(total, on="OverTime")
    grp["Percentage"] = grp["Count"] / grp["Total"] * 100

    fig = px.bar(
        grp[grp["Attrition"] == "Yes"],
        x="OverTime", y="Percentage",
        color="OverTime",
        color_discrete_map={"Yes": DANGER, "No": SUCCESS},
        title="OverTime vs Attrition Rate (%)",
        template=TEMPLATE,
        text=grp[grp["Attrition"] == "Yes"]["Percentage"].round(1).astype(str) + "%",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def age_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Age", color="Attrition",
        color_discrete_map=COLOR_MAP,
        barmode="overlay",
        opacity=0.75,
        title="Age Distribution by Attrition",
        template=TEMPLATE,
        nbins=30,
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def job_satisfaction_attrition(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["JobSatisfaction", "Attrition"]).size().reset_index(name="Count")
    sat_labels = {1: "1-Low", 2: "2-Medium", 3: "3-High", 4: "4-Very High"}
    grp["Satisfaction"] = grp["JobSatisfaction"].map(sat_labels)
    fig = px.bar(
        grp, x="Satisfaction", y="Count", color="Attrition",
        color_discrete_map=COLOR_MAP,
        barmode="group",
        title="Job Satisfaction vs Attrition",
        template=TEMPLATE,
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    # Use only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    # Encode Attrition for correlation if it's still string
    if "Attrition" in df.columns and df["Attrition"].dtype == object:
        numeric_df = numeric_df.copy()
        numeric_df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    corr = numeric_df.corr()
    # Focus on top correlated features with Attrition
    if "Attrition" in corr.columns:
        top_features = (
            corr["Attrition"].abs().sort_values(ascending=False).head(15).index.tolist()
        )
        corr = corr.loc[top_features, top_features]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, DANGER], [0.5, "#1E293B"], [1, ACCENT]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
        showscale=True,
    ))
    fig.update_layout(
        title="Correlation Heatmap (Top Features vs Attrition)",
        template=TEMPLATE,
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def years_at_company_attrition(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="YearsAtCompany", color="Attrition",
        color_discrete_map=COLOR_MAP,
        barmode="overlay",
        opacity=0.75,
        title="Years at Company vs Attrition",
        template=TEMPLATE,
        nbins=25,
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def job_role_attrition(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["JobRole", "Attrition"]).size().reset_index(name="Count")
    total = df.groupby("JobRole").size().reset_index(name="Total")
    grp = grp.merge(total, on="JobRole")
    grp["Rate"] = grp["Count"] / grp["Total"] * 100

    yes_grp = grp[grp["Attrition"] == "Yes"].sort_values("Rate", ascending=True)
    fig = go.Figure(go.Bar(
        x=yes_grp["Rate"],
        y=yes_grp["JobRole"],
        orientation="h",
        marker=dict(
            color=yes_grp["Rate"],
            colorscale=[[0, SUCCESS], [0.5, WARNING], [1, DANGER]],
            showscale=True,
        ),
        text=yes_grp["Rate"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig.update_layout(
        title="Attrition Rate by Job Role (%)",
        xaxis_title="Attrition Rate (%)",
        template=TEMPLATE,
        height=420,
        margin=dict(l=20, r=80, t=50, b=20),
    )
    return fig


def marital_status_attrition(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby(["MaritalStatus", "Attrition"]).size().reset_index(name="Count")
    fig = px.bar(
        grp, x="MaritalStatus", y="Count", color="Attrition",
        color_discrete_map=COLOR_MAP,
        barmode="group",
        title="Marital Status vs Attrition",
        template=TEMPLATE,
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig
