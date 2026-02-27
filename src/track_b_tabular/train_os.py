"""
Standalone training script for Azure ML command jobs and pipeline steps.
Trains a tabular classifier on Contoso service orders to predict repair type
(RepairType: Overhaul vs Preventive).
"""

import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from preprocess_os import load_and_clean_os, prepare_train_test


MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    ),
}


def train_and_evaluate(model_name, X_train, X_test, y_train, y_test):
    model = MODELS[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return model, y_pred, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="logistic_regression",
                        choices=list(MODELS.keys()))
    parser.add_argument("--max-categories", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mlflow.autolog(log_models=False)

    df = load_and_clean_os(args.input_data)
    X_train, X_test, y_train, y_test, preprocessor, _, _ = prepare_train_test(
        df, test_size=args.test_size, max_categories=args.max_categories,
    )

    mlflow.log_param("model_type", args.model_name)
    mlflow.log_param("max_categories", args.max_categories)
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("n_test_samples", X_test.shape[0])
    mlflow.log_param("n_features", X_train.shape[1])

    model, y_pred, metrics = train_and_evaluate(
        args.model_name, X_train, X_test, y_train, y_test,
    )

    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    report = classification_report(
        y_test, y_pred, target_names=["Preventive", "Overhaul"],
    )
    print(report)
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    model_path = os.path.join(args.output_dir, "model.joblib")
    preprocessor_path = os.path.join(args.output_dir, "preprocessor.joblib")
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=None,
    )
    mlflow.log_artifact(preprocessor_path, artifact_path="model")

    print(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()
