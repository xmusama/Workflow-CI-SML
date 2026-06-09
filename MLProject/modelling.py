from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA_PATH = "dataset_preprocessing/processed.csv"


def to_snake_case(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def ensure_processed_dataset(data_path: Path) -> Path:
    if data_path.exists():
        return data_path

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    if "target" not in df.columns:
        df["target"] = data.target

    df.columns = [to_snake_case(c) if c != "target" else "target" for c in df.columns]

    x_features = df.drop(columns=["target"])
    y_target = df["target"].astype(int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_features)
    processed = pd.DataFrame(x_scaled, columns=x_features.columns)
    processed["target"] = y_target.values

    processed.to_csv(data_path, index=False)
    return data_path


def train(data_path: Path, output_dir: Path, n_estimators: int, max_depth: int) -> None:
    data_path = ensure_processed_dataset(data_path)
    df = pd.read_csv(data_path)

    x_features = df.drop(columns=["target"])
    y_target = df["target"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.2, random_state=42, stratify=y_target
    )

    if not os.getenv("MLFLOW_EXPERIMENT_ID") and not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment("workflow_ci_training")

    def log_and_save() -> None:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        mlflow.sklearn.log_model(model, "model")

        output_dir.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(model, output_dir)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    run_id = os.getenv("MLFLOW_RUN_ID")
    if mlflow.active_run() is None:
        with mlflow.start_run(run_id=run_id):
            log_and_save()
    else:
        log_and_save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default="artifacts/model")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()

    train(
        Path(args.data_path), Path(args.output_dir), args.n_estimators, args.max_depth
    )


if __name__ == "__main__":
    main()
