"""
Utils Module
Shared helpers for metrics, plotting, and risk categorisation.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from sklearn.tree import export_text


PLOTLY_TEMPLATE = "plotly_dark"
PRIMARY   = "#7C3AED"
ACCENT    = "#06B6D4"
DANGER    = "#EF4444"
SUCCESS   = "#10B981"
WARNING   = "#F59E0B"


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Attrition", "Attrition"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[[0, "#0F172A"], [0.5, PRIMARY], [1.0, ACCENT]],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
            showscale=False,
        )
    )
    fig.update_layout(
        title=f"{model_name} — Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template=PLOTLY_TEMPLATE,
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_roc_curve(model, X_test, y_test, model_name: str) -> go.Figure:
    fig = go.Figure()
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            return fig

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"AUC = {roc_auc:.3f}",
            line=dict(color=ACCENT, width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="#475569", dash="dash"),
        ))
        fig.update_layout(
            title=f"{model_name} — ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template=PLOTLY_TEMPLATE,
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(x=0.6, y=0.1),
        )
    except Exception:
        pass
    return fig


def plot_feature_importance(model, feature_names: list, model_name: str) -> go.Figure:
    try:
        importances = model.feature_importances_
    except AttributeError:
        try:
            importances = np.abs(model.coef_[0])
        except Exception:
            return go.Figure()

    idx = np.argsort(importances)[-20:]
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker=dict(
            color=importances[idx],
            colorscale=[[0, PRIMARY], [1, ACCENT]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        title=f"{model_name} — Feature Importance",
        xaxis_title="Importance",
        template=PLOTLY_TEMPLATE,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def get_decision_rules(dt_model, feature_names: list, max_depth: int = 4) -> str:
    """Export decision tree rules as a readable string."""
    try:
        rules = export_text(dt_model, feature_names=feature_names, max_depth=max_depth)
        return rules
    except Exception as e:
        return f"Could not extract rules: {e}"


def risk_category(prob: float) -> tuple[str, str]:
    """Return (label, color) for an attrition probability."""
    if prob >= 0.65:
        return "🔴 HIGH RISK", DANGER
    elif prob >= 0.35:
        return "🟡 MEDIUM RISK", WARNING
    else:
        return "🟢 LOW RISK", SUCCESS


def model_accuracy_comparison(results: dict) -> go.Figure:
    """Bar chart comparing all model accuracies."""
    names = list(results.keys())
    accs  = [results[n]["metrics"]["Accuracy"] for n in names]
    colors = [ACCENT if a == max(accs) else PRIMARY for a in accs]

    fig = go.Figure(go.Bar(
        x=accs,
        y=names,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{a:.1f}%" for a in accs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Model Accuracy Comparison",
        xaxis_title="Accuracy (%)",
        xaxis=dict(range=[0, 105]),
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(l=20, r=60, t=50, b=20),
    )
    return fig
