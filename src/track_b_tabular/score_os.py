"""
Scoring script for the Contoso service orders repair-type classifier.
Deployed as an Azure ML Managed Online Endpoint.
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
import mlflow.sklearn

logger = logging.getLogger(__name__)


def init():
    """Called once when the endpoint starts. Loads model and preprocessor."""
    global model, preprocessor

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model = mlflow.sklearn.load_model(model_dir)
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    logger.info("Model and preprocessor loaded successfully.")


def run(raw_data: str) -> str:
    """
    Called for each scoring request.

    Expected input format:
    {
        "data": [
            {
                "EquipmentModel": "EX200",
                "JobCode": "PM",
                "ServiceCenter": "1001",
                "QtyOrdered": 4.0,
                "month": 12,
                "quarter": 4,
                "day_of_week": 0
            }
        ]
    }
    """
    try:
        request = json.loads(raw_data)
        input_df = pd.DataFrame(request["data"])
        input_df["ServiceCenter"] = input_df["ServiceCenter"].astype(str)

        X = preprocessor.transform(input_df)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        label_map = {1: "Overhaul", 0: "Preventive"}
        results = []
        for i in range(len(predictions)):
            prob_overhaul = float(probabilities[i][1])
            results.append({
                "predicted_repair_type": label_map[int(predictions[i])],
                "overhaul_probability": round(prob_overhaul, 4),
                "confidence": (
                    "high" if prob_overhaul > 0.8 or prob_overhaul < 0.2
                    else "medium" if prob_overhaul > 0.6 or prob_overhaul < 0.4
                    else "low"
                ),
                "input": request["data"][i],
            })

        return json.dumps({"results": results}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Scoring error: {e}")
        return json.dumps({"error": str(e)})
