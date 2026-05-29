"""
Preprocessor Module
Handles feature engineering, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

# Columns that carry no useful variance
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]

# Categorical columns to label-encode
CATEGORICAL_COLS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
]


@st.cache_data(show_spinner="Preprocessing data...")
def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, scaler, label_encoders, df_encoded
    """
    df = df.copy()

    # Drop useless columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Encode target
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Label-encode categoricals & store encoders for inverse use
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    feature_names = [c for c in df.columns if c != "Attrition"]
    X = df[feature_names].values
    y = df["Attrition"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        feature_names, scaler,
        label_encoders, df
    )


def encode_single_employee(employee_dict: dict, feature_names: list,
                            label_encoders: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Encode a single employee's raw feature dict into a scaled feature vector
    ready for model.predict().
    """
    row = {}
    for col in feature_names:
        val = employee_dict.get(col, 0)
        if col in label_encoders:
            le = label_encoders[col]
            known = list(le.classes_)
            if str(val) in known:
                val = le.transform([str(val)])[0]
            else:
                val = 0
        row[col] = val

    vec = np.array([[row[c] for c in feature_names]], dtype=float)
    vec_scaled = scaler.transform(vec)
    return vec_scaled
