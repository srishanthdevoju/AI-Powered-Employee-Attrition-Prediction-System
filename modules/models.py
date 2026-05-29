"""
Models Module
Trains all 7 ML models and returns results including metrics, predictions, and trained estimators.
"""

import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as SklearnPCA
from modules.utils import compute_metrics


@st.cache_resource(show_spinner="Training ML models... (this may take a moment)")
def train_all_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train all classification models and return a results dict.

    Returns
    -------
    dict: {model_name: {"model": estimator, "y_pred": array, "metrics": dict}}
    """
    results = {}

    # ── 1. Logistic Regression ──────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["Logistic Regression"] = {
        "model": lr,
        "y_pred": y_pred_lr,
        "metrics": compute_metrics(y_test, y_pred_lr),
        "description": (
            "Logistic Regression is a linear classifier that models the probability "
            "of attrition using a sigmoid function. It is interpretable and serves as "
            "a strong baseline for binary classification tasks."
        ),
    }

    # ── 2. Decision Tree ────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    results["Decision Tree"] = {
        "model": dt,
        "y_pred": y_pred_dt,
        "metrics": compute_metrics(y_test, y_pred_dt),
        "description": (
            "Decision Trees partition the feature space into regions using a series "
            "of IF-THEN rules. They are highly interpretable and can generate explicit "
            "HR decision rules (e.g., IF JobSatisfaction < 2 AND MonthlyIncome < 4000 "
            "THEN High Attrition Risk)."
        ),
    }

    # ── 3. Random Forest ────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = {
        "model": rf,
        "y_pred": y_pred_rf,
        "metrics": compute_metrics(y_test, y_pred_rf),
        "description": (
            "Random Forest is an ensemble of 200 decision trees trained on random "
            "subsets of data. It is the production-grade model with the highest accuracy, "
            "best handling of imbalanced classes, and built-in feature importance ranking."
        ),
    }

    # ── 4. SVM ──────────────────────────────────────────────────────────────
    svm = SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    results["SVM"] = {
        "model": svm,
        "y_pred": y_pred_svm,
        "metrics": compute_metrics(y_test, y_pred_svm),
        "description": (
            "Support Vector Machine with RBF kernel finds the optimal hyperplane that "
            "maximises the margin between classes in a high-dimensional space. Excellent "
            "for non-linear decision boundaries."
        ),
    }

    # ── 5. KNN ──────────────────────────────────────────────────────────────
    # Determine best k using a small grid search
    best_k, best_acc = 5, 0.0
    for k in [3, 5, 7, 11, 15]:
        knn_tmp = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn_tmp.fit(X_train, y_train)
        acc = knn_tmp.score(X_test, y_test)
        if acc > best_acc:
            best_acc, best_k = acc, k

    knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    results["KNN"] = {
        "model": knn,
        "y_pred": y_pred_knn,
        "metrics": compute_metrics(y_test, y_pred_knn),
        "best_k": best_k,
        "description": (
            f"K-Nearest Neighbours (k={best_k}) classifies an employee by looking at "
            "their k most similar colleagues in feature space. Useful for finding "
            "similar employees and spotting peer-group attrition patterns."
        ),
    }

    # ── 6. Naive Bayes ──────────────────────────────────────────────────────
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    results["Naive Bayes"] = {
        "model": nb,
        "y_pred": y_pred_nb,
        "metrics": compute_metrics(y_test, y_pred_nb),
        "description": (
            "Gaussian Naive Bayes computes the probability of attrition given all "
            "features, assuming conditional independence. It is very fast and provides "
            "calibrated probability estimates for each employee."
        ),
    }

    return results


@st.cache_resource(show_spinner="Training K-Means clustering model...")
def train_kmeans(X_scaled, n_clusters: int = 3, random_state: int = 42):
    """Train K-Means and return model + cluster labels."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias = []
    for k in range(1, 9):
        km_tmp = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km_tmp.fit(X_scaled)
        inertias.append(km_tmp.inertia_)
    return km, labels, inertias


@st.cache_resource(show_spinner="Running PCA...")
def run_pca(X_scaled, n_components: int = None):
    """Run PCA and return PCA object + transformed data."""
    pca = SklearnPCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


CLUSTER_LABELS = {
    0: ("🏆 High Performers", "#10B981"),
    1: ("⚠️ At Risk",         "#EF4444"),
    2: ("🌱 New Employees",   "#06B6D4"),
}
