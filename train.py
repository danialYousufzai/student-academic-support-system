
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    precision_recall_curve
)

from features import DATA_FILE, MODEL_FEATURES, add_features, make_target


def train_and_save(dataset_path=None, output_dir=None):

    base_dir = Path(__file__).parent

    if dataset_path is None:
        dataset_path = base_dir / DATA_FILE

    if output_dir is None:
        output_dir = base_dir

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    print("Loading student dataset...")

    raw = pd.read_csv(dataset_path)

    if "GradeClass" not in raw.columns:
        raise ValueError("Dataset must contain GradeClass.")

    data = add_features(raw)

    X = data[MODEL_FEATURES]
    y = make_target(data)

    if y.nunique() != 2:
        raise ValueError(
            "Training requires both target classes."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("Training Logistic Regression...")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=10,
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Find the best high-risk threshold using
    # out-of-fold training predictions.

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_train,
        oof_probabilities
    )

    f2_scores = (
        5 * precision[:-1] * recall[:-1]
        / np.clip(4 * precision[:-1] + recall[:-1], 1e-9, None)
    )

    high_threshold = float(
        thresholds[np.argmax(f2_scores)]
    )

    medium_threshold = high_threshold / 2

    print("Fitting final model...")

    model.fit(X_train, y_train)

    # Evaluate on test data

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= high_threshold
    ).astype(int)

    metrics = {
        "roc_auc": round(
            float(roc_auc_score(y_test, probabilities)), 4
        ),
        "pr_auc": round(
            float(average_precision_score(y_test, probabilities)), 4
        ),
        "recall": round(
            float(recall_score(y_test, predictions, zero_division=0)), 4
        ),
        "precision": round(
            float(precision_score(y_test, predictions, zero_division=0)), 4
        )
    }

    meta = {
        "model": "Logistic Regression",
        "best_params": {"classifier__C": 10},
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "target": "needs_support = GradeClass == 4 (F, GPA < 2.0)",
        "high_threshold": round(high_threshold, 4),
        "medium_threshold": round(medium_threshold, 4),
        "metrics": metrics,
        "features": MODEL_FEATURES
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save newly trained model

    joblib.dump(
        model,
        output_dir / "model.joblib"
    )

    # Save model information

    with open(output_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save demo cohort

    demo = raw.loc[
        X_test.index,
        ["StudentID"] + list(X.columns)[:0]
    ] if False else raw.loc[
        X_test.index,
        ["StudentID"] + [
            "Age", "Absences", "StudyTimeWeekly",
            "Tutoring", "ParentalSupport",
            "Extracurricular", "Sports", "Music",
            "Volunteering", "GradeClass"
        ]
    ]

    demo.head(60).to_csv(
        output_dir / "sample_upload.csv",
        index=False
    )

    print("Model training completed!")
    print("Model saved successfully.")
    print("Metrics:", metrics)

    return meta


if __name__ == "__main__":
    train_and_save()
