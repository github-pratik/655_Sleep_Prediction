#!/usr/bin/env python3
"""
Train baseline models (Logistic Regression, Random Forest) on fatigue prediction.
Requires dataset/train.csv and dataset/test.csv produced by step4_time_split.py
Outputs:
  models/logreg.pkl
  models/rf.pkl
  reports/metrics.json
  reports/confusion_matrix.png
  reports/feature_importance.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

TARGET_COL = "fatigue_label"


def load_data(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]
    return feature_cols, X_train, X_test, y_train, y_test


def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return le, y_train_enc, y_test_enc


def build_preprocessor(feature_cols):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), feature_cols)
        ],
        remainder="drop",
    )


def train_and_eval(model_name, model, preprocessor, X_train, y_train, X_test, y_test):
    clf = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return clf, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}


def plot_confusion(cm, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(model_pipeline, feature_cols, out_path: Path):
    # Extract feature importances from RF inside the pipeline
    model = model_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    sorted_feats = [feature_cols[i] for i in order]
    sorted_imps = importances[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(sorted_imps)), sorted_imps[::-1])
    ax.set_yticks(range(len(sorted_feats)))
    ax.set_yticklabels(sorted_feats[::-1])
    ax.set_xlabel("Importance")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=Path("dataset/train.csv"), type=Path)
    parser.add_argument("--test", default=Path("dataset/test.csv"), type=Path)
    args = parser.parse_args()

    feature_cols, X_train, X_test, y_train_raw, y_test_raw = load_data(args.train, args.test)
    le, y_train, y_test = encode_labels(y_train_raw, y_test_raw)

    preprocessor = build_preprocessor(feature_cols)

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
    }

    metrics = {}
    trained = {}

    for name, model in models.items():
        clf, m = train_and_eval(name, model, preprocessor, X_train, y_train, X_test, y_test)
        metrics[name] = {k: float(v) if k != "cm" else m["cm"].tolist() for k, v in m.items()}
        trained[name] = clf
        joblib.dump(clf, Path(f"models/{name}.pkl"))

    # Plots using RF confusion and importances
    rf_cm = np.array(metrics["rf"]["cm"])
    labels = le.classes_.tolist()
    plot_confusion(rf_cm, labels, Path("reports/confusion_matrix.png"))
    plot_feature_importance(trained["rf"], feature_cols, Path("reports/feature_importance.png"))

    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    best = max(metrics.items(), key=lambda kv: kv[1]["f1"])
    print(f"Best model: {best[0]} with F1={best[1]['f1']:.3f}")


if __name__ == "__main__":
    main()
