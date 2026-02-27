"""
Generate synthetic datasets for the Azure ML MLOps Workshop.

Creates two CSV files in the data/ directory:
  - inspections_dataset.csv   (Track A: text classification)
  - service_orders_dataset.csv (Track B: tabular classification)

Usage:
    pip install faker pandas numpy
    python generate_sample_data.py
"""

import os
import random

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Track A: Inspection comments (text classification)
# ---------------------------------------------------------------------------

LEAD_COMMENTS = [
    "Hydraulic cylinder rod leak detected on boom assembly",
    "Excessive wear on brake pads, replacement recommended",
    "Increasing cylinder pipe leak, pipe replacement recommended",
    "Operational damage to cylinder, urgent replacement required",
    "Turbocharger showing early signs of failure, recommend inspection",
    "Cracked weld on frame rail near engine mount",
    "Abnormal vibration in final drive, bearings may need replacement",
    "Oil contamination in hydraulic system, filter change overdue",
    "Track link pin wear exceeding tolerance, chain replacement needed",
    "Cooling system pressure loss detected, radiator inspection needed",
    "Steering cylinder seal leaking, replacement required",
    "Transmission slipping under load, clutch pack inspection needed",
    "Bucket teeth heavily worn, full set replacement recommended",
    "Engine blow-by exceeding limits, overhaul may be required",
    "Alternator output dropping, electrical system check needed",
    "Fuel injector nozzle fouling detected on cylinder 3",
    "Swing bearing play detected, tightening or replacement needed",
    "Exhaust temperature high on bank A, possible turbo issue",
    "Undercarriage idler cracked, safety concern flagged",
    "Air filter restriction alarm triggered repeatedly",
]

NON_LEAD_COMMENTS = [
    "Not applicable",
    "System functionality test performed, satisfactory result",
    "Equipment parked at staging area awaiting tires",
    "Routine visual inspection completed, no issues found",
    "All readings within normal operating range",
    "Scheduled service completed per maintenance plan",
    "No abnormalities detected during walkthrough",
    "Fluid levels checked and topped off",
    "Equipment operating normally after filter change",
    "Greasing completed on all lubrication points",
    "No action required at this time",
    "Inspection deferred - equipment not available",
    "Pre-shift check completed, machine cleared for operation",
    "Battery charge level normal, connections secure",
    "Tire pressure within specification",
    "No leaks observed during pressure test",
    "Calibration verified, instruments within tolerance",
    "Safety devices tested and functioning correctly",
    "Belts and hoses in good condition",
    "Operator reports no concerns",
]


def generate_inspections(n_rows=10500):
    """Generate synthetic inspection comments dataset."""
    lead_ratio = 0.25
    n_leads = int(n_rows * lead_ratio)
    n_non_leads = n_rows - n_leads

    rows = []
    for _ in range(n_leads):
        comment = random.choice(LEAD_COMMENTS)
        noise = f" Unit {fake.bothify('??-####')}." if random.random() > 0.5 else ""
        prob = round(random.uniform(0.55, 0.99), 2)
        rows.append({
            "comment": comment + noise,
            "is_lead_opportunity": 1,
            "confidence": "high" if prob > 0.8 else "medium" if prob > 0.6 else "low",
        })

    for _ in range(n_non_leads):
        comment = random.choice(NON_LEAD_COMMENTS)
        noise = f" Ref {fake.bothify('##-???')}." if random.random() > 0.3 else ""
        prob = round(random.uniform(0.01, 0.45), 2)
        rows.append({
            "comment": comment + noise,
            "is_lead_opportunity": 0,
            "confidence": "high" if prob < 0.2 else "medium" if prob < 0.4 else "low",
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Track B: Service orders (tabular classification)
# ---------------------------------------------------------------------------

EQUIPMENT_MODELS = ["EX100", "EX200", "DZ300", "LH400", "TR500",
                    "GD150", "WL250", "SK350", "HT450", "CR550"]
JOB_CODES = ["PM", "RP", "CM", "IN", "EM", "OV"]
SERVICE_CENTERS = ["1001", "1002", "1003", "2001", "2002",
                   "3001", "3002", "4001", "4002", "5001"]


def generate_service_orders(n_rows=425745):
    """Generate synthetic service orders dataset."""
    order_ids = np.arange(100000, 100000 + n_rows)

    equipment_models = np.random.choice(EQUIPMENT_MODELS, size=n_rows, p=[
        0.15, 0.18, 0.12, 0.10, 0.08, 0.10, 0.07, 0.08, 0.06, 0.06
    ])

    job_codes = np.random.choice(JOB_CODES, size=n_rows, p=[
        0.30, 0.25, 0.20, 0.10, 0.10, 0.05
    ])

    service_centers = np.random.choice(SERVICE_CENTERS, size=n_rows)

    qty_ordered = np.round(
        np.random.lognormal(mean=0.5, sigma=1.0, size=n_rows).clip(1, 200), 1
    )

    start_date = pd.Timestamp("2019-01-01")
    end_date = pd.Timestamp("2024-12-31")
    date_range_days = (end_date - start_date).days
    random_days = np.random.randint(0, date_range_days, size=n_rows)
    dates = pd.to_datetime(start_date) + pd.to_timedelta(random_days, unit="D")
    order_request_dates = dates.strftime("%Y%m%d").astype(int)

    overhaul_probs = np.zeros(n_rows)
    overhaul_probs += np.where(np.isin(equipment_models, ["EX200", "LH400", "TR500"]), 0.15, 0.0)
    overhaul_probs += np.where(np.isin(job_codes, ["OV", "EM"]), 0.25, 0.0)
    overhaul_probs += np.where(qty_ordered > 10, 0.10, 0.0)
    overhaul_probs += 0.10
    overhaul_probs = overhaul_probs.clip(0, 1)

    repair_types = np.where(
        np.random.random(n_rows) < overhaul_probs, "Overhaul", "Preventive"
    )

    null_mask_date = np.random.random(n_rows) < 0.02
    null_mask_qty = np.random.random(n_rows) < 0.01

    df = pd.DataFrame({
        "OrderID": order_ids,
        "EquipmentModel": equipment_models,
        "JobCode": job_codes,
        "ServiceCenter": service_centers,
        "QtyOrdered": qty_ordered,
        "OrderRequestDate": order_request_dates,
        "RepairType": repair_types,
    })

    df.loc[null_mask_date, "OrderRequestDate"] = np.nan
    df.loc[null_mask_qty, "QtyOrdered"] = np.nan

    return df


if __name__ == "__main__":
    print("Generating inspection comments dataset...")
    df_inspections = generate_inspections()
    inspections_path = os.path.join(DATA_DIR, "inspections_dataset.csv")
    df_inspections.to_csv(inspections_path, index=False)
    print(f"  Saved: {inspections_path} ({len(df_inspections):,} rows)")

    print("Generating service orders dataset...")
    df_orders = generate_service_orders()
    orders_path = os.path.join(DATA_DIR, "service_orders_dataset.csv")
    df_orders.to_csv(orders_path, index=False)
    print(f"  Saved: {orders_path} ({len(df_orders):,} rows)")

    print("\nDone! Datasets are ready in the data/ directory.")
    print(f"  Inspections: {df_inspections.shape}")
    print(f"  Service orders: {df_orders.shape}")
