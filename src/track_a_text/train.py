"""
Standalone training script for Azure ML command jobs and pipeline steps.
Trains a text classifier on Contoso inspection comments to predict sales lead
opportunities (is_lead_opportunity).
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
    confusion_matrix,
)

from preprocess import load_and_clean_inspections, prepare_train_test


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

    if model_name == "gradient_boosting":
        X_train = X_train.toarray()
        X_test = X_test.toarray()

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
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mlflow.autolog(log_models=False)

    df = load_and_clean_inspections(args.input_data)
    X_train, X_test, y_train, y_test, vectorizer, _, _ = prepare_train_test(
        df, test_size=args.test_size, max_features=args.max_features
    )

    mlflow.log_param("model_type", args.model_name)
    mlflow.log_param("max_features", args.max_features)
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("n_test_samples", X_test.shape[0])

    model, y_pred, metrics = train_and_evaluate(
        args.model_name, X_train, X_test, y_train, y_test
    )

    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    report = classification_report(y_test, y_pred, target_names=["No Lead", "Lead"])
    print(report)
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    model_path = os.path.join(args.output_dir, "model.joblib")
    vectorizer_path = os.path.join(args.output_dir, "vectorizer.joblib")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=None,
    )
    mlflow.log_artifact(vectorizer_path, artifact_path="model")

    print(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()
