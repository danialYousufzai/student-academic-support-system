"""Shared feature engineering + business rules (notebook, trainer and Streamlit app all use this)."""
import numpy as np
import pandas as pd

DATA_FILE = "student_performance_data.csv"

# Raw columns the model is allowed to see (attendance, study habits, support context)
BASE_FEATURES = ["Age", "Absences", "StudyTimeWeekly", "Tutoring", "ParentalSupport",
                 "Extracurricular", "Sports", "Music", "Volunteering"]
ENGINEERED = ["activities_count", "high_absence", "low_study", "absence_study_ratio"]
MODEL_FEATURES = BASE_FEATURES + ENGINEERED
REQUIRED_COLUMNS = BASE_FEATURES

# Never used as model inputs: only for the fairness audit
AUDIT_COLUMNS = ["Gender", "Ethnicity", "ParentalEducation"]

HIGH_ABSENCE = 15      # absences per year considered high
LOW_STUDY = 5          # study hours / week considered low

SUPPORT_LABELS = {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very high"}


def make_target(df: pd.DataFrame) -> pd.Series:
    """needs_support = 1 when GradeClass == 4 (F, i.e. GPA < 2.0)."""
    return (df["GradeClass"] == 4).astype(int)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["activities_count"] = d[["Extracurricular", "Sports", "Music", "Volunteering"]].sum(axis=1)
    d["high_absence"] = (d["Absences"] >= HIGH_ABSENCE).astype(int)
    d["low_study"] = (d["StudyTimeWeekly"] < LOW_STUDY).astype(int)
    d["absence_study_ratio"] = d["Absences"] / (d["StudyTimeWeekly"] + 1)
    return d


def risk_tier(p, high_thr, medium_thr):
    return np.where(p >= high_thr, "High", np.where(p >= medium_thr, "Medium", "Low"))


def explain_row(r) -> str:
    """Plain-language key factors for a student (NaN-safe)."""
    out = []
    a, s = r.get("Absences", np.nan), r.get("StudyTimeWeekly", np.nan)
    if a >= HIGH_ABSENCE:
        out.append(f"{int(a)} absences (high)")
    elif a >= 10:
        out.append(f"{int(a)} absences (rising)")
    if s < LOW_STUDY:
        out.append(f"Only {s:.1f} study hrs/week")
    elif s < 8:
        out.append(f"Moderate study time ({s:.1f} hrs/week)")
    if r.get("ParentalSupport", np.nan) <= 1:
        out.append("Low parental support")
    if r.get("Tutoring", np.nan) == 0 and (a >= HIGH_ABSENCE or s < LOW_STUDY):
        out.append("No tutoring in place")
    if r.get("activities_count", 1) == 0:
        out.append("No extracurricular engagement")
    return "; ".join(out) if out else "No single dominant factor"
