
"""
Student Academic-Support Early-Warning System
Run locally: streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import (
    DATA_FILE,
    MODEL_FEATURES,
    REQUIRED_COLUMNS,
    SUPPORT_LABELS,
    add_features,
    explain_row,
    make_target,
    risk_tier
)

st.set_page_config(
    page_title="Student Support Early-Warning",
    page_icon="🎓",
    layout="wide"
)

HERE = Path(__file__).parent

TIER_ICON = {
    "High": "🔴",
    "Medium": "🟠",
    "Low": "🟢"
}


# ---------------------------------------------------------
# MODEL LOADING - UPDATED
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")

def load_artifacts():

    model_path = HERE / "model.joblib"
    meta_path = HERE / "model_meta.json"

    # First, try loading the existing model
    if model_path.exists() and meta_path.exists():

        try:
            model = joblib.load(model_path)

            with open(meta_path, "r") as f:
                meta = json.load(f)

            return model, meta

        except Exception as error:

            st.warning(
                f"Saved model could not be loaded. "
                f"Attempting to retrain. Details: {error}"
            )

    else:

        st.info(
            "Saved model files are missing. "
            "Training a new model..."
        )

    # If loading fails, train a new model
    try:

        from train import train_and_save

        dataset_path = HERE / DATA_FILE

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Training dataset not found: {dataset_path}"
            )

        meta = train_and_save(dataset_path, HERE)

        # Load newly trained model
        model = joblib.load(model_path)

        # Load metadata if training did not return it
        if meta is None:
            with open(meta_path, "r") as f:
                meta = json.load(f)

        return model, meta

    except Exception as error:

        st.error(
            "Unable to load or train the machine learning model."
        )

        st.exception(error)

        st.stop()


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model, meta = load_artifacts()


# ---------------------------------------------------------
# SCORING FUNCTION
# ---------------------------------------------------------

def score(df: pd.DataFrame, high_thr: float, med_thr: float):

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    d = df.copy()

    for c in REQUIRED_COLUMNS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = add_features(d)

    d["risk_probability"] = model.predict_proba(
        d[MODEL_FEATURES]
    )[:, 1].round(3)

    d["risk_tier"] = risk_tier(
        d["risk_probability"].values,
        high_thr,
        med_thr
    )

    d["key_factors"] = d.apply(
        explain_row,
        axis=1
    )

    return d


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("🎓 Early-Warning")

    st.caption(
        "Flags students likely to need academic support "
        "(predicted GPA < 2.0 / grade F) from attendance, "
        "study habits and support factors."
    )

    st.subheader("Triage settings")

    high_thr = st.slider(
        "High-risk threshold",
        0.05,
        0.90,
        float(meta["high_threshold"]),
        0.01,
        help="At/above this probability a student is High."
    )

    med_thr = st.slider(
        "Medium-risk threshold",
        0.02,
        float(high_thr),
        min(
            float(meta["medium_threshold"]),
            float(high_thr)
        ),
        0.01
    )

    st.divider()

    st.caption(
        f"Model: **{meta['model']}**  \n"
        f"Trained: {meta['trained_at']}"
    )

    st.warning(
        "Decision-support only. A flag should start a "
        "supportive conversation, never an automatic "
        "decision about a student."
    )


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab_single, tab_batch, tab_about = st.tabs([
    "👤 Single student",
    "📂 Class / cohort upload",
    "ℹ️ About the model"
])


# ---------------------------------------------------------
# SINGLE STUDENT
# ---------------------------------------------------------

with tab_single:

    st.subheader("Enter a student's data")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("**Attendance & study**")

        age = st.slider("Age", 15, 18, 16)

        absences = st.slider(
            "Absences this year",
            0, 30, 10
        )

        study = st.slider(
            "Study time (hours / week)",
            0.0, 20.0, 10.0, 0.5
        )

    with c2:

        st.markdown("**Support**")

        tutoring = st.radio(
            "Receiving tutoring?",
            ["No", "Yes"],
            horizontal=True
        )

        support = st.select_slider(
            "Parental support",
            options=list(SUPPORT_LABELS),
            value=2,
            format_func=lambda k: SUPPORT_LABELS[k]
        )

    with c3:

        st.markdown("**Activities**")

        extra = st.checkbox("Extracurricular")

        sports = st.checkbox("Sports")

        music = st.checkbox("Music")

        vol = st.checkbox("Volunteering")

    if st.button("Assess student", type="primary"):

        row = pd.DataFrame([{
            "Age": age,
            "Absences": absences,
            "StudyTimeWeekly": study,
            "Tutoring": int(tutoring == "Yes"),
            "ParentalSupport": support,
            "Extracurricular": int(extra),
            "Sports": int(sports),
            "Music": int(music),
            "Volunteering": int(vol)
        }])

        try:

            res = score(row, high_thr, med_thr).iloc[0]

            a, b = st.columns([1, 2])

            a.metric(
                "Risk of needing support",
                f"{res.risk_probability:.0%}"
            )

            a.markdown(
                f"### {TIER_ICON[res.risk_tier]} "
                f"{res.risk_tier} priority"
            )

            b.progress(float(res.risk_probability))

            b.markdown(
                "**Key factors:** " + res.key_factors
            )

            if res.risk_tier != "Low":

                b.info(
                    "Suggested next step: a short check-in "
                    "to understand barriers to attendance, "
                    "a study-habit plan, and tutoring or "
                    "catch-up sessions."
                )

        except Exception as e:

            st.error(f"Assessment error: {e}")


# ---------------------------------------------------------
# BATCH UPLOAD
# ---------------------------------------------------------

with tab_batch:

    st.subheader("Score a whole class or cohort")

    st.markdown(
        "Upload a CSV with these columns "
        "(`StudentID` and other columns are kept; "
        f"if `GradeClass` is present, accuracy is shown): "
        f"`{', '.join(REQUIRED_COLUMNS)}`"
    )

    sample = load_sample() if False else None

    sample_path = HERE / "sample_upload.csv"

    if sample_path.exists():

        sample = pd.read_csv(sample_path)

    else:

        sample = None

    if sample is not None:

        st.download_button(
            "⬇️ Download sample CSV template",
            sample.to_csv(index=False).encode(),
            "sample_upload.csv",
            "text/csv"
        )

    up = st.file_uploader(
        "Upload CSV",
        type="csv"
    )

    use_demo = st.checkbox(
        "…or use the built-in demo cohort",
        value=up is None and sample is not None
    )

    df_in = (
        pd.read_csv(up)
        if up is not None
        else (sample if use_demo else None)
    )

    if df_in is not None:

        try:

            out = score(df_in, high_thr, med_thr)

        except ValueError as e:

            st.error(str(e))

        else:

            if "StudentID" not in out.columns:

                out.insert(
                    0,
                    "StudentID",
                    [f"ROW{i+1}" for i in range(len(out))]
                )

            counts = (
                out["risk_tier"]
                .value_counts()
                .reindex(["High", "Medium", "Low"])
                .fillna(0)
                .astype(int)
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Students", len(out))

            m2.metric("🔴 High", counts["High"])

            m3.metric("🟠 Medium", counts["Medium"])

            m4.metric("🟢 Low", counts["Low"])

            st.bar_chart(counts)

            if "GradeClass" in out.columns:

                actual = make_target(out)

                flagged = (
                    out["risk_probability"] >= high_thr
                ).astype(int)

                rec = (
                    ((flagged == 1) & (actual == 1)).sum()
                    / max(actual.sum(), 1)
                )

                prec = (
                    ((flagged == 1) & (actual == 1)).sum()
                    / max(flagged.sum(), 1)
                )

                st.success(
                    f"Back-test on your labels: "
                    f"**recall {rec:.0%}** "
                    f"(at-risk students caught), "
                    f"**precision {prec:.0%}** "
                    f"(flags that were correct)."
                )

            tiers = st.multiselect(
                "Show tiers",
                ["High", "Medium", "Low"],
                default=["High", "Medium"]
            )

            show = (
                out[out["risk_tier"].isin(tiers)]
                .sort_values(
                    "risk_probability",
                    ascending=False
                )
            )

            cols = [
                "StudentID",
                "risk_tier",
                "risk_probability",
                "key_factors",
                "Absences",
                "StudyTimeWeekly",
                "Tutoring",
                "ParentalSupport"
            ]

            st.dataframe(
                show[cols],
                hide_index=True,
                column_config={
                    "risk_probability":
                    st.column_config.ProgressColumn(
                        "risk_probability",
                        min_value=0,
                        max_value=1,
                        format="%.2f"
                    )
                }
            )

            st.download_button(
                "⬇️ Download full results (CSV)",
                out.to_csv(index=False).encode(),
                "support_risk_results.csv",
                "text/csv"
            )


# ---------------------------------------------------------
# ABOUT THE MODEL
# ---------------------------------------------------------

with tab_about:

    st.subheader("How it works")

    m = meta["metrics"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("ROC-AUC (hold-out)", m["roc_auc"])

    c2.metric("PR-AUC", m["pr_auc"])

    c3.metric("Recall @ threshold", m["recall"])

    c4.metric("Precision @ threshold", m["precision"])

    st.markdown(f"""
    **Goal** – flag students likely to finish with a failing
    grade (**{meta.get('target', 'needs support')}**) using
    attendance, study habits and support context, so help
    can be offered **early**.

    **Design choices**

    - Recall-oriented threshold (F2): missing a struggling
      student is costlier than a supportive check-in.
    - `GPA` is **not** a model input (it is the outcome);
      gender, ethnicity and parental education are **not**
      inputs either – they are used only in a fairness audit.
    - Every flag comes with plain-language key factors
      so staff can verify it.

    **Limitations & responsible use**

    - The Kaggle dataset is synthetic and ~7% of its grade
      labels are noisy; real-world accuracy will differ.
      Retrain and validate on your own anonymised data,
      and monitor performance by sub-group.
    - Predictions are probabilities, not diagnoses;
      use them alongside teacher judgement.
    - Keep data anonymised; never upload names or other
      direct identifiers.
    """)

    with st.expander("Model metadata"):

        st.json(meta)
