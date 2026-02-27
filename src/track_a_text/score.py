"""
Scoring script for the Contoso inspection lead classifier.
Deployed as an Azure ML Managed Online Endpoint.
"""

import os
import json
import joblib
import logging
import re
import numpy as np
import mlflow.sklearn

logger = logging.getLogger(__name__)


def init():
    """Called once when the endpoint starts. Loads model and vectorizer."""
    global model, vectorizer

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model = mlflow.sklearn.load_model(model_dir)
    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
    logger.info("Model and vectorizer loaded successfully.")


def clean_text(text: str) -> str:
    """Mirror the preprocessing from training."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def run(raw_data: str) -> str:
    """
    Called for each scoring request.

    Expected input format:
    {
        "data": [
            {"comment": "Hydraulic cylinder rod leak detected"},
            {"comment": "Not applicable"}
        ]
    }
    """
    try:
        request = json.loads(raw_data)
        comments = [item["comment"] for item in request["data"]]
        cleaned = [clean_text(c) for c in comments]

        X = vectorizer.transform(cleaned)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        results = []
        for i, comment in enumerate(comments):
            results.append({
                "comment": comment,
                "is_lead_opportunity": bool(predictions[i]),
                "probability": float(probabilities[i][1]),
                "confidence": (
                    "high" if probabilities[i][1] > 0.8 or probabilities[i][1] < 0.2
                    else "medium" if probabilities[i][1] > 0.6 or probabilities[i][1] < 0.4
                    else "low"
                ),
            })

        return json.dumps({"results": results}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Scoring error: {e}")
        return json.dumps({"error": str(e)})
