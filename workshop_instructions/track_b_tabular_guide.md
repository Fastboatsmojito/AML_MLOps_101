# Track B: Predict Service Order Repair Type (Tabular)

## Self-Paced Take-Home Guide

This guide walks you through building an end-to-end MLOps pipeline on Azure Machine Learning using **tabular (structured) data**. You will train a classifier that predicts whether a service order is an **Overhaul** or **Preventive** repair based on structured fields like equipment model, job code, and quantity ordered.

Track B follows the same MLOps patterns as Track A (text classification) but with tabular data. If you completed Track A during the workshop, you already have Azure resources set up and can skip Part 1.

**No prior Azure ML experience is required.** Follow each step in order.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Part 1: Azure Setup (Skip if you completed Track A)](#2-part-1-azure-setup-skip-if-you-completed-track-a)
3. [Part 2: Open the Notebooks](#3-part-2-open-the-notebooks)
4. [Part 3: Notebook 01b - Data Versioning (Tabular)](#4-part-3-notebook-01b---data-versioning-tabular-15-min)
5. [Part 4: Notebook 02b - Experiment Tracking (Tabular)](#5-part-4-notebook-02b---experiment-tracking-tabular-25-min)
6. [Part 5: Notebook 03b - Model Registration (Tabular)](#6-part-5-notebook-03b---model-registration-tabular-10-min)
7. [Part 6: Notebook 04b - Model Deployment (Tabular)](#7-part-6-notebook-04b---model-deployment-tabular-30-min)
8. [Part 7: Notebook 05b - Model Monitoring (Tabular)](#8-part-7-notebook-05b---model-monitoring-tabular-15-min)
9. [Part 8: Notebook 06b - ML Pipelines (Tabular)](#9-part-8-notebook-06b---ml-pipelines-tabular-15-min)
10. [Cleanup](#10-cleanup)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

- [ ] Access to the **shared Azure subscription**
- [ ] **Contributor** role on the subscription
- [ ] Your **initials** ready (e.g., `jd` for Jane Doe)
- [ ] If you completed Track A: your existing Azure ML workspace and compute instance
- [ ] If you did NOT complete Track A: follow Part 1 below to set everything up

---

## 2. Part 1: Azure Setup (Skip if you completed Track A)

If you already completed Track A, your resource group, workspace, compute instance, and compute cluster are already set up. Skip to [Part 2](#3-part-2-open-the-notebooks).

If starting fresh, follow these steps:

### Step 2.1: Find your Subscription ID

1. Go to [portal.azure.com](https://portal.azure.com)
2. Search for **Subscriptions** in the search bar
3. Click on the shared subscription and copy the **Subscription ID**

### Step 2.2: Create a Resource Group

1. Search for **Resource groups** in the portal
2. Click **+ Create**
3. Fill in:
   - **Subscription**: Select the shared subscription
   - **Resource group name**: `rg-aml-workshop-{YOUR_INITIALS}` (e.g., `rg-aml-workshop-jd`)
   - **Region**: `East US 2`
4. Click **Review + create**, then **Create**

### Step 2.3: Create an Azure ML Workspace

1. Search for **Azure Machine Learning** in the portal
2. Click **+ Create** > **New workspace**
3. Fill in:
   - **Subscription**: Select the shared subscription
   - **Resource group**: `rg-aml-workshop-{YOUR_INITIALS}`
   - **Workspace name**: `aml-workshop-{YOUR_INITIALS}` (e.g., `aml-workshop-jd`)
   - **Region**: `East US 2`
4. Click **Review + create**, then **Create** (takes 2-4 minutes)
5. Click **Go to resource**, then **Launch studio**

### Step 2.4: Create a Compute Instance

1. In Azure ML Studio, go to **Compute** > **Compute instances**
2. Click **+ New**
3. Set **Compute name**: `notebook-{YOUR_INITIALS}` and **Size**: `Standard_DS3_v2`
4. Click **Create** (takes 2-3 minutes)

### Step 2.5: Create a Compute Cluster

1. Go to **Compute** > **Compute clusters** > **+ New**
2. Size: `Standard_DS3_v2`, Name: `cpu-cluster`, Min nodes: `0`, Max nodes: `2`
3. Click **Create**

### Step 2.6: Clone the Repository

1. In Azure ML Studio, go to **Notebooks** > open a **Terminal**
2. Run:

```bash
cd ~/cloudfiles/code/Users/$(whoami)
git clone https://github.com/Fastboatsmojito/AML_MLOps_101.git
cd AML_MLOps_101
pip install -r requirements.txt -q
python generate_sample_data.py
```

The last command generates the synthetic datasets in the `data/` directory (~10 seconds).

---

## 3. Part 2: Open the Notebooks

1. In Azure ML Studio, click **Notebooks** in the left sidebar
2. Navigate to: **Users > {your-username} > AML_MLOps_101 > notebooks > track_b_tabular**
3. You should see notebooks `01b` through `06b`
4. When opening a notebook, select kernel **Python 3.10 - SDK v2** (or **Python 3.10**)

### Updating Hardcoded Values

**Every notebook** contains hardcoded `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME` values that you must update. Look for the `<<<< CHANGE THIS` comments in the first code cell of each notebook.

```python
SUBSCRIPTION_ID = "your-subscription-id"         # From Step 2.1
RESOURCE_GROUP = "rg-aml-workshop-jd"             # From Step 2.2
WORKSPACE_NAME = "aml-workshop-jd"                # From Step 2.3
```

---

## 4. Part 3: Notebook 01b - Data Versioning (Tabular) ~15 min

**Open**: `notebooks/track_b_tabular/01b_data_versioning_tabular.ipynb`

### What this notebook does
Registers the service orders dataset (`service_orders_dataset.csv` — 425,745 rows) as a versioned Data Asset. It profiles the raw data, cleans it (engineers date features, handles nulls, creates labels), and registers the cleaned version as v2.

### Before you run
Update `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME` (look for `<<<< CHANGE THIS`).

### Run it
Run each cell top to bottom. You should see:
- Data asset `service-orders` registered as v1 (raw) and v2 (cleaned)
- Data profiling statistics (row counts, column types, null rates)
- Label distribution: how many Overhaul vs Preventive orders

### What to look for in Azure ML Studio
Go to **Data** in the left sidebar. You should see `service-orders` with 2 versions.

### What to do if it fails

| Error | Fix |
|-------|-----|
| `FileNotFoundError` for data file | Verify the file exists: run `ls ../../data/` in a terminal. You should see `service_orders_dataset.csv`. If not, re-clone the repo. |
| `UserErrorException` during upload | Storage auth issue. Restart your compute instance and try again. |

---

## 5. Part 4: Notebook 02b - Experiment Tracking (Tabular) ~25 min

**Open**: `notebooks/track_b_tabular/02b_experiment_tracking_tabular.ipynb`

### What this notebook does
Trains 4 classifiers on the tabular service orders data:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Logistic Regression (different regularization)

The key difference from Track A: instead of TF-IDF text vectors (5,000 sparse features), this uses **one-hot encoded categoricals + scaled numerics** (~70 dense features).

### Before you run
Update `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME`.

### Run it
Each model trains in 5-15 seconds (tabular data is much faster than text). You should see:
- Confusion matrices for each model (with "Preventive" and "Overhaul" labels)
- A comparison table ranked by F1 score
- A bar chart comparing all metrics

### What to look for in Azure ML Studio
Go to **Jobs** > `contoso-repair-classifier` experiment. Compare this with Track A's `contoso-lead-classifier` experiment if you completed it — same MLflow interface, different data.

### What to do if it fails

| Error | Fix |
|-------|-----|
| `UnicodeEncodeError` | Run: `import os; os.environ["PYTHONIOENCODING"] = "utf-8"` before the failing cell. |
| `ValueError` in preprocessing | The data might have unexpected values. Make sure Notebook 01b ran successfully (the cleaning step handles edge cases). |

---

## 6. Part 5: Notebook 03b - Model Registration (Tabular) ~10 min

**Open**: `notebooks/track_b_tabular/03b_model_registration_tabular.ipynb`

### What this notebook does
Finds the best tabular model by F1 score and registers it as `contoso-repair-classifier` in the Model Registry. If you completed Track A, the registry will show both models side by side.

### Before you run
Update `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME`.

### Run it
You should see:
- Best run identified with its F1 score and model type
- Model registered as `contoso-repair-classifier` v1
- If Track A was completed: both `contoso-lead-classifier` and `contoso-repair-classifier` listed

### What to look for in Azure ML Studio
Go to **Models**. If you did both tracks, you will see two models — one for text, one for tabular. Click each to compare their tags and lineage.

### What to do if it fails

| Error | Fix |
|-------|-----|
| `IndexError: list index out of range` | No completed runs found. Re-run Notebook 02b first. |
| `"(not yet registered — complete Track A first)"` message | This is normal if you skipped Track A. It's just a comparison check and does not affect Track B. |

---

## 7. Part 6: Notebook 04b - Model Deployment (Tabular) ~30 min

**Open**: `notebooks/track_b_tabular/04b_model_deployment_tabular.ipynb`

### What this notebook does
Deploys the tabular model as a live REST API endpoint. The scoring script (`score_os.py`) accepts structured JSON with fields like `EquipmentModel`, `JobCode`, `ServiceCenter`, `QtyOrdered`, and date features — then returns a prediction.

### Before you run

1. Update `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME`
2. **IMPORTANT**: Update `ENDPOINT_NAME` — add your initials (e.g., `contoso-repair-classifier-jd`). Endpoint names must be unique across all participants in the same region. Look for the `<<<< CHANGE THIS` comment.

### Run it

- **Cell 3** (Create endpoint): ~30 seconds
- **Cell 5** (Create deployment): **5-10 minutes**. Be patient — a container is being built.
- **Cell 6** (Route traffic): Sets 100% to blue deployment.
- **Cell 8** (Test): Sends 5 structured service order records and gets predictions.

Expected output:
```
  EX200    | Overhaul             | prob_overhaul: 0.82 | confidence: high
  DZ300    | Preventive           | prob_overhaul: 0.23 | confidence: high
```

### What to look for in Azure ML Studio
Go to **Endpoints**. If you did both tracks, you should see two endpoints:
- `contoso-lead-classifier-{initials}` — accepts text input
- `contoso-repair-classifier-{initials}` — accepts structured JSON

### What to do if it fails

| Error | Fix |
|-------|-----|
| `Endpoint name already exists` | Add more characters to make it unique (e.g., `contoso-repair-classifier-jd-2`). |
| Deployment takes >15 minutes | Check status in Studio > Endpoints. As long as it shows "Updating", it is still working. |
| `AuthorizationFailure` | Storage policy issue. Ask the facilitator for help. |
| Test returns error | Check deployment logs in Studio > Endpoints > your endpoint > Logs. The deployment may still be provisioning. |

---

## 8. Part 7: Notebook 05b - Model Monitoring (Tabular) ~15 min

**Open**: `notebooks/track_b_tabular/05b_model_monitoring_tabular.ipynb`

### What this notebook does
Sets up monitoring for the tabular model. Tabular monitoring is more intuitive than text monitoring because you can see concrete drift signals like:
- "The distribution of equipment models shifted — we're seeing more LH400 orders"
- "Average quantity ordered increased by 40%"
- "A new job code appeared that wasn't in training data"

### Before you run

1. Update `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME`
2. **IMPORTANT**: Update `ENDPOINT_NAME` to match the exact name you used in Notebook 04b (with your initials). Look for `<<<< CHANGE THIS`.
3. **IMPORTANT**: Update the `emails` field to your own email address. Look for `<<<< CHANGE THIS TO YOUR EMAIL ADDRESS`.

### Run it
- Training baseline analysis runs instantly
- Drift visualization charts are displayed inline (these are simulated scenarios showing what drift looks like)
- Monitoring schedule is created

### What to look for in Azure ML Studio
Go to **Monitoring**. You should see `contoso-repair-monitor` scheduled.

### What to do if it fails

| Error | Fix |
|-------|-----|
| `ResourceNotFoundError` for endpoint | Your `ENDPOINT_NAME` doesn't match what you used in Notebook 04b. They must be identical. |

---

## 9. Part 8: Notebook 06b - ML Pipelines (Tabular) ~15 min

**Open**: `notebooks/track_b_tabular/06b_pipeline_definition_tabular.ipynb`

### What this notebook does
Defines a reusable training pipeline for the tabular model. The structure is nearly identical to Track A's pipeline — you just swap the data input and scripts, demonstrating that pipelines are reusable patterns.

### Before you run
Update `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME`.

### Run it
- **Cell 3**: Registers the training environment (may take 2-3 minutes for Docker build on first run)
- **Cell 8**: Submits the pipeline job — click the Studio URL to watch it run
- **Cell 9**: Streams output (~5-10 minutes)
- **Cell 11**: Submits two more pipeline runs with different model types

### What to look for in Azure ML Studio
Go to **Jobs** > `contoso-repair-classifier-pipeline`. Click on a run to see the pipeline DAG visualization. If you completed Track A, compare the two pipeline experiments — same structure, different data.

### What to do if it fails

| Error | Fix |
|-------|-----|
| `ResourceNotFoundError` for `cpu-cluster` | Create it: Compute > Compute clusters > + New > Name: `cpu-cluster`, Size: `Standard_DS3_v2`, Min: 0, Max: 2. |
| Pipeline stuck in "Preparing" | The cluster is scaling from 0 to 1 node. Wait 3-5 minutes. |
| `ImageBuildFailure` | Environment Docker build failed. Retry by re-running the cell. |

---

## 10. Cleanup

**IMPORTANT**: Endpoints cost money while running. Clean up when done.

### Delete the Endpoint

In the last cell of Notebook 04b, uncomment and run:

```python
ml_client.online_endpoints.begin_delete(name=ENDPOINT_NAME).result()
print(f"Endpoint '{ENDPOINT_NAME}' deleted.")
```

### Delete the Monitoring Schedule

In Azure ML Studio:
1. Go to **Monitoring**
2. Click on `contoso-repair-monitor`
3. Click **Delete**

### Delete the Compute Instance

1. Go to **Compute** > **Compute instances**
2. Select your compute instance
3. Click **Delete**

### Delete All Resources

To remove everything at once:
1. Go to [portal.azure.com](https://portal.azure.com) > **Resource groups**
2. Click on `rg-aml-workshop-{YOUR_INITIALS}`
3. Click **Delete resource group**
4. Type the name to confirm and click **Delete**

---

## 11. Troubleshooting

### Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Wrong credentials** | `ResourceNotFoundError` or authentication errors | Look for `<<<< CHANGE THIS` comments in every notebook. Update `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME` to match your Azure resources. |
| **Kernel dies** | Notebook stops responding | Go to **Compute**, restart your compute instance, then re-open the notebook. |
| **Module not found** | `ModuleNotFoundError` | Run `pip install -r requirements.txt` in a terminal, then restart the kernel. |
| **Token expired** | `HttpResponseError` or timeouts | Restart the kernel and re-run the connection cell (first code cell). |
| **File not found** | Can't find data or scripts | Verify the repo structure: run `ls ~/cloudfiles/code/Users/$(whoami)/AML_MLOps_101/data/` in a terminal. |
| **Quota exceeded** | `OperationNotAllowed` | Try a smaller VM size, or ask the facilitator to request a quota increase. |
| **Endpoint name conflict** | `Endpoint already exists` | Append more characters to your endpoint name to make it unique. |

### Key Differences from Track A

| Aspect | Track A (Text) | Track B (Tabular) |
|--------|---------------|-------------------|
| **Dataset** | 10,500 inspection comments | 425,745 service orders |
| **Features** | TF-IDF vectors (5,000 sparse) | One-hot + scaled numerics (~70 dense) |
| **Preprocessing** | Text cleaning, vectorization | Date engineering, encoding, scaling |
| **Scoring input** | `{"data": [{"comment": "..."}]}` | `{"data": [{"EquipmentModel": "EX200", ...}]}` |
| **Drift signals** | Abstract (TF-IDF distribution) | Concrete (equipment model %, quantities) |
| **MLOps infrastructure** | Identical | Identical |

The takeaway: **MLOps is data-type agnostic.** The same Azure ML features — Data Assets, MLflow, Model Registry, Managed Endpoints, Monitoring, Pipelines — work identically for text and tabular problems.
