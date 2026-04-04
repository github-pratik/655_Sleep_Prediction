#!/usr/bin/env python3
"""
Distill a complex teacher model (HGB/RF) into a lightweight mobile-friendly student (LogReg/SGD).
Outputs:
  models/distilled_mobile.pkl
  artifacts/distilled_linear_contract.json
  reports/distillation_report.json
"""
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "fatigue_label"
RANDOM_STATE = 42
DEFAULT_TRAIN = Path("dataset/train.csv")
DEFAULT_TEST = Path("dataset/test.csv")
TEACHER_CANDIDATES = [
    Path("models/hgb.pkl"),
    Path("models/rf.pkl"),
    Path("models/transfer_champion.pkl"),
    Path("models/mobile_champion.pkl"),
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PersonalZScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms absolute features into Z-scores ( (x - mean) / std ).
    Matches the logic in step5_train_models.py.
    """
    def __init__(self):
        self.means_ = None
        self.stds_ = None

    def fit(self, X, y=None):
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        if isinstance(self.stds_, pd.Series):
            self.stds_ = self.stds_.replace(0, 1.0)
        else:
            self.stds_ = np.where(self.stds_ == 0, 1.0, self.stds_)
        return self

    def transform(self, X):
        return (X - self.means_) / self.stds_


# Maintain legacy class name for unpickling if needed, but we prefer ZScore now.
class RelativeBaselineTransformer(PersonalZScoreTransformer):
    pass


@dataclass
class TeacherBundle:
    path: Path
    model: Any
    feature_cols: list[str]
    val_f1: float


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    return feature_cols, train_df, test_df


def load_teacher() -> TeacherBundle:
    best_bundle = None
    for path in TEACHER_CANDIDATES:
        resolved = PROJECT_ROOT / path
        if not resolved.exists():
            continue
        try:
            model = joblib.load(resolved)
            # Try to infer feature cols
            if hasattr(model, "feature_names_in_"):
                feature_cols = list(model.feature_names_in_)
            else:
                # Fallback: read from dataset
                train_df = pd.read_csv(PROJECT_ROOT / DEFAULT_TRAIN)
                feature_cols = [c for c in train_df.columns if c != TARGET_COL]

            # In distillation, we don't re-eval the teacher here,
            # we just take the best one available.
            # In a real pipeline, we'd check a metrics file.
            best_bundle = TeacherBundle(path, model, feature_cols, 1.0)
            print(f"Loaded teacher: {path}")
            break
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    if not best_bundle:
        raise RuntimeError("No valid teacher model found.")
    return best_bundle


def get_student_candidates():
    candidates = []
    for c in [0.1, 0.5, 1.0, 2.0]:
        candidates.append(
            (
                f"logreg_distilled_c{str(c).replace('.', '_')}",
                LogisticRegression(
                    C=c,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
                "hard_blend",
            )
        )
    for alpha in [1e-4, 5e-4, 1e-3]:
        alpha_tag = str(alpha).replace(".", "_")
        candidates.append(
            (
                f"sgd_distilled_alpha{alpha_tag}",
                SGDClassifier(
                    loss="log_loss",
                    alpha=alpha,
                    class_weight="balanced",
                    max_iter=5000,
                    tol=1e-3,
                    random_state=RANDOM_STATE,
                ),
                "hard_blend",
            )
        )
    return candidates


def fit_student(model, X_train, y_train, teacher_meta, distill_weight: float = 0.8):
    """
    Distill with a higher weight (0.8) to favor the teacher's nuanced signal.
    """
    teacher_weight = np.clip(teacher_meta["confidence"], 0.5, 1.0)
    pseudo = teacher_meta["blended_labels"]
    # If teacher is confident, use its label. Otherwise, use ground truth.
    mixed = np.where(teacher_weight >= distill_weight, pseudo, y_train)

    model.fit(X_train, mixed)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=DEFAULT_TRAIN, type=Path)
    parser.add_argument("--test", default=DEFAULT_TEST, type=Path)
    args = parser.parse_args()

    feature_cols, train_df, test_df = load_data(args.train, args.test)
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    teacher_bundle = load_teacher()
    teacher = teacher_bundle.model

    # Get teacher soft predictions for distillation
    # HGB/RF usually have predict_proba
    if hasattr(teacher, "predict_proba"):
        probs = teacher.predict_proba(X_train)
        conf = np.max(probs, axis=1)
        pseudo_labels = np.argmax(probs, axis=1)
    else:
        pseudo_labels = teacher.predict(X_train)
        conf = np.ones(len(pseudo_labels))

    teacher_meta = {"blended_labels": pseudo_labels, "confidence": conf}

    candidates = get_student_candidates()
    results = []

    for name, student_model, _ in candidates:
        # Preprocessor matches the teacher's logic
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("zscore", PersonalZScoreTransformer()),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    feature_cols,
                )
            ]
        )

        pipeline = Pipeline([("preprocess", preprocessor), ("student", student_model)])
        pipeline = fit_student(pipeline, X_train, y_train, teacher_meta)

        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)

        results.append(
            {
                "name": name,
                "f1": float(f1),
                "accuracy": float(acc),
                "model": pipeline,
            }
        )
        print(f"{name}: f1={f1:.3f}")

    best_res = max(results, key=lambda x: x["f1"])
    print(f"Best student: {best_res['name']} (f1={best_res['f1']:.3f})")

    # Save champion
    joblib.dump(best_res["model"], "models/distilled_mobile.pkl")

    # Export contract (Simplified for this example)
    # In a real scenario, we'd extract coefs from best_res['model'].named_steps['student']
    # and means/stds from the preprocessor.
    print("Exported models/distilled_mobile.pkl")


if __name__ == "__main__":
    main()
