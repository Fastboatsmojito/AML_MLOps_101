# Track A: Classify Inspection Comments as Sales Leads

## Step-by-Step Workshop Guide (~3 hours)

This guide walks you through building an end-to-end MLOps pipeline on Azure Machine Learning. You will train a text classifier that reads inspection comments and predicts whether they represent a sales lead opportunity.

**No prior Azure ML experience is required.** Follow each step in order.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Part 1: Azure Setup (Portal)](#2-part-1-azure-setup-portal-20-min)
3. [Part 2: Clone the Repo and Launch Notebooks](#3-part-2-clone-the-repo-and-launch-notebooks-10-min)
4. [Part 3: Notebook 00 - Setup & Config](#4-part-3-notebook-00---setup--config-5-min)
5. [Part 4: Notebook 01 - Data Versioning](#5-part-4-notebook-01---data-versioning-15-min)
6. [Part 5: Notebook 02 - Experiment Tracking](#6-part-5-notebook-02---experiment-tracking-30-min)
7. [Part 6: Notebook 03 - Model Registration](#7-part-6-notebook-03---model-registration-15-min)
8. [Part 7: Notebook 04 - Model Deployment](#8-part-7-notebook-04---model-deployment-30-min)
9. [Part 8: Notebook 05 - Model Monitoring](#9-part-8-notebook-05---model-monitoring-20-min)
10. [Part 9: Notebook 06 - ML Pipelines](#10-part-9-notebook-06---ml-pipelines-20-min)
11. [Cleanup](#11-cleanup)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

Before you begin, make sure you have:

- [ ] Access to the **shared Azure subscription** (you should be able to sign in at [portal.azure.com](https://portal.azure.com))
- [ ] **Contributor** role on the subscription (ask your admin if unsure)
- [ ] A web browser (Edge or Chrome recommended)
- [ ] Your **initials** ready (e.g., `jd` for Jane Doe) — you will append these to resource names to avoid conflicts with other participants

---

## 2. Part 1: Azure Setup (Portal) ~20 min

We need to create three Azure resources: a **Resource Group**, an **Azure ML Workspace**, and a **Compute Instance**. All of this is done in the Azure Portal.

### Step 2.1: Find your Subscription ID

1. Go to [portal.azure.com](https://portal.azure.com)
2. In the search bar at the top, type **Subscriptions** and click on it
3. You should see the shared subscription listed. Click on it.
4. Copy the **Subscription ID** (a long string like `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`). You will need this later.

### Step 2.2: Create a Resource Group

A Resource Group is a container that holds all your Azure resources.

1. In the Azure Portal search bar, type **Resource groups** and click on it
2. Click **+ Create**
3. Fill in:
   - **Subscription**: Select the shared subscription
   - **Resource group name**: `rg-aml-workshop-{YOUR_INITIALS}` (e.g., `rg-aml-workshop-jd`)
   - **Region**: `East US 2`
4. Click **Review + create**, then **Create**
5. Wait for the deployment to complete (a few seconds)

### Step 2.3: Create an Azure ML Workspace

The Azure ML Workspace is the central hub for all ML activities.

1. In the Azure Portal search bar, type **Azure Machine Learning** and click on it
2. Click **+ Create** > **New workspace**
3. Fill in:
   - **Subscription**: Select the shared subscription
   - **Resource group**: Select `rg-aml-workshop-{YOUR_INITIALS}` (the one you just created)
   - **Workspace name**: `aml-workshop-{YOUR_INITIALS}` (e.g., `aml-workshop-jd`)
   - **Region**: `East US 2` (must match your resource group)
4. Leave all other settings as default
5. Click **Review + create**, then **Create**
6. **Wait for deployment** — this takes 2-4 minutes. You will see "Your deployment is complete" when done.
7. Click **Go to resource**

### Step 2.4: Launch Azure ML Studio

1. On the workspace overview page, click **Launch studio** (blue button)
2. This opens **Azure ML Studio** (ml.azure.com) — this is where you will work for the rest of the workshop

### Step 2.5: Create a Compute Instance

A Compute Instance is your cloud-based development machine where you run notebooks.

1. In Azure ML Studio, click **Compute** in the left sidebar (under **Manage**)
2. Click the **Compute instances** tab
3. Click **+ New**
4. Fill in:
   - **Compute name**: `notebook-{YOUR_INITIALS}` (e.g., `notebook-jd`). Must be unique.
   - **Virtual machine size**: Select `Standard_DS3_v2` (4 cores, 14 GB RAM). If not visible, click "Select from all options" and search for it.
5. Click **Create**
6. Wait for the status to change from "Creating" to **Running** (2-3 minutes)

### Step 2.6: Create a Compute Cluster

A Compute Cluster is used for running ML pipelines (Notebook 06).

1. Stay in **Compute**, click the **Compute clusters** tab
2. Click **+ New**
3. Fill in:
   - **Virtual machine size**: `Standard_DS3_v2`
   - Click **Next**
   - **Compute name**: `cpu-cluster`
   - **Minimum number of nodes**: `0`
   - **Maximum number of nodes**: `2`
   - **Idle seconds before scale down**: `300`
4. Click **Create**

---

## 3. Part 2: Clone the Repo and Launch Notebooks ~10 min

### Step 3.1: Open a Terminal on your Compute Instance

1. In Azure ML Studio, click **Notebooks** in the left sidebar
2. Click **Terminal** (the `>_` icon at the top) to open a terminal session
3. If prompted, select your compute instance (`notebook-{YOUR_INITIALS}`)

### Step 3.2: Clone the Repository

In the terminal, run:

```bash
cd ~/cloudfiles/code/Users/$(whoami)
git clone https://github.com/Fastboatsmojito/AML_MLOps_101.git
```

### Step 3.3: Install Dependencies and Generate Data

```bash
cd AML_MLOps_101
pip install -r requirements.txt -q
```

This installs all required Python packages (Azure ML SDK, MLflow, scikit-learn, etc.). Takes about 1-2 minutes.

### Step 3.4: Generate the Sample Datasets

The datasets are generated synthetically. Run the generation script:

```bash
python generate_sample_data.py
```

This creates two files in the `data/` directory:
- `inspections_dataset.csv` (10,500 rows — inspection comments for text classification)
- `service_orders_dataset.csv` (425,745 rows — service orders for tabular classification)

The script takes about 10-15 seconds.

### Step 3.5: Navigate to the Notebooks

1. In the left sidebar of Azure ML Studio, click **Notebooks**
2. In the file browser, navigate to: **Users > {your-username} > AML_MLOps_101 > notebooks**
3. You should see `00_setup_and_config.ipynb` and the `track_a_text/` folder

### Step 3.6: Select the Right Kernel

When you open any notebook:
1. You will see a kernel selector in the top-right. Click on it.
2. Select **Python 3.10 - SDK v2** (or **Python 3.10** if SDK v2 is not listed)
3. Make sure your compute instance is selected as the compute target

---

## 4. Part 3: Notebook 00 - Setup & Config ~5 min

**Open**: `notebooks/00_setup_and_config.ipynb`

### What this notebook does
Connects to your Azure ML workspace and verifies that your compute resources exist.

### Before you run

**IMPORTANT**: In the code cell that contains `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME`, you **must** update these values to match the resources you created in Part 1:

```python
SUBSCRIPTION_ID = "your-subscription-id-here"       # The ID you copied in Step 2.1
RESOURCE_GROUP = "rg-aml-workshop-jd"                # The resource group you created in Step 2.2
WORKSPACE_NAME = "aml-workshop-jd"                   # The workspace you created in Step 2.3
```

Look for the `<<<< CHANGE THIS` comments in the code — they mark every value you need to update.

### Run it

Run each cell from top to bottom (Shift+Enter). You should see:
- "Connected to workspace: aml-workshop-{your-initials}"
- Your compute cluster listed

### What to do if it fails

| Error | Fix |
|-------|-----|
| `DefaultAzureCredential failed` | Make sure you are running on an Azure ML compute instance (not locally). The compute instance inherits your Azure identity automatically. |
| `ResourceNotFoundError` for workspace | Double-check your `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME` values. They must exactly match what you created in the portal. |
| `pip install` errors | Try running `pip install azure-ai-ml azure-identity mlflow azureml-mlflow` manually in a terminal cell. |

---

## 5. Part 4: Notebook 01 - Data Versioning ~15 min

**Open**: `notebooks/track_a_text/01_data_versioning.ipynb`

### What this notebook does
Registers the inspection comments dataset as a versioned **Data Asset** in Azure ML. You will create two versions: the raw data (v1) and a cleaned version (v2). This is how Azure ML tracks data lineage.

### Before you run
Update the `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME` values (look for `<<<< CHANGE THIS` comments).

### Run it
Run each cell from top to bottom. You should see:
- Data asset `classified-inspections` registered as v1 and v2
- A comparison between the raw and cleaned versions

### What to look for in Azure ML Studio
After running, go to **Data** in the left sidebar of Azure ML Studio. You should see `classified-inspections` with 2 versions.

### What to do if it fails

| Error | Fix |
|-------|-----|
| `FileNotFoundError` for data file | The path `../../data/inspections_dataset.csv` is relative to the notebook location. Make sure you cloned the repo correctly and the `data/` folder contains the data file. Run `ls ../../data/` in a terminal to verify. |
| `UserErrorException` during data upload | This usually means a storage authentication issue. Try restarting your compute instance from the **Compute** page and re-running. |

---

## 6. Part 5: Notebook 02 - Experiment Tracking ~30 min

**Open**: `notebooks/track_a_text/02_experiment_tracking.ipynb`

### What this notebook does
This is the core ML notebook. It:
1. Sets up MLflow experiment tracking (Azure ML uses MLflow natively)
2. Loads and preprocesses the inspection comments using TF-IDF vectorization
3. Trains 4 models: Logistic Regression, Random Forest, Gradient Boosting, and another Logistic Regression variant
4. Logs metrics (accuracy, precision, recall, F1), confusion matrices, and classification reports to MLflow
5. Compares all models programmatically

### Before you run
Update the `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME` values (look for `<<<< CHANGE THIS` comments).

### Run it
Run each cell from top to bottom. Each model takes 10-30 seconds to train. You will see:
- Confusion matrix plots displayed inline
- Metrics printed after each model
- A final comparison table showing all models ranked by F1 score

### What to look for in Azure ML Studio
Go to **Jobs** in the left sidebar. Click on the experiment `contoso-lead-classifier`. You should see 4 completed runs. Click on any run to explore:
- **Metrics** tab: accuracy, precision, recall, F1
- **Images** tab: confusion matrix
- **Outputs + logs** tab: classification report text file

Select 2+ runs and click **Compare** to see a side-by-side comparison.

### What to do if it fails

| Error | Fix |
|-------|-----|
| `UnicodeEncodeError` | Run this in a cell before the failing cell: `import os; os.environ["PYTHONIOENCODING"] = "utf-8"` |
| `AttributeError: 'MLClient' object has no attribute 'tracking_uri'` | This was already fixed in the repo. Make sure you have the latest version (re-clone if needed). |
| Training takes very long | This is normal for Gradient Boosting on sparse matrices. Wait ~30-60 seconds. |

---

## 7. Part 6: Notebook 03 - Model Registration ~15 min

**Open**: `notebooks/track_a_text/03_model_registration.ipynb`

### What this notebook does
Finds the best model from your experiments (by F1 score) and registers it in the **Azure ML Model Registry**. The registry is a governed catalog of models with versioning, tags, and lineage.

### Before you run
Update the `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME` values.

### Run it
You should see:
- The best run identified with its F1 score
- Model registered as `contoso-lead-classifier` v1
- Tags showing data version, region, and training metrics

### What to look for in Azure ML Studio
Go to **Models** in the left sidebar. Click on `contoso-lead-classifier` to see:
- Version history
- Tags (data asset, F1 score, region)
- Link back to the training run (full lineage)

### What to do if it fails

| Error | Fix |
|-------|-----|
| `IndexError: list index out of range` | No completed runs found. Go back and make sure Notebook 02 ran successfully with at least one model trained. |

---

## 8. Part 7: Notebook 04 - Model Deployment ~30 min

**Open**: `notebooks/track_a_text/04_model_deployment.ipynb`

### What this notebook does
Deploys the registered model as a live REST API endpoint. This is where your ML model starts delivering business value — any application can send inspection comments and get predictions back.

### Before you run

1. Update `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME`
2. **IMPORTANT**: Update `ENDPOINT_NAME` — add your initials (e.g., `contoso-lead-classifier-jd`). Endpoint names must be unique across all participants in the same Azure region. Look for the `<<<< CHANGE THIS` comment.

### Run it

- **Cell 3** (Create endpoint): Takes ~30 seconds. Creates the endpoint URL.
- **Cell 5** (Create deployment): **This takes 5-10 minutes.** A container is being built with your model, scoring script, and dependencies. Be patient — do not re-run this cell.
- **Cell 6** (Route traffic): Sets 100% traffic to the blue deployment.
- **Cell 8** (Test): Sends 7 inspection comments and gets predictions back.

You should see output like:
```
  [LEAD] (prob: 0.87, conf: high) Hydraulic cylinder rod leak detected
  [----] (prob: 0.12, conf: high) Not applicable
```

### What to look for in Azure ML Studio
Go to **Endpoints** in the left sidebar. Click on your endpoint to see:
- **Details** tab: scoring URI, auth keys
- **Test** tab: paste JSON to test interactively
- **Logs** tab: real-time container logs

### What to do if it fails

| Error | Fix |
|-------|-----|
| `Endpoint name already exists` | Another participant used the same name. Add more characters to make it unique (e.g., `contoso-lead-classifier-jd-2`). |
| Deployment takes >15 minutes | This is sometimes normal on shared subscriptions. Check the deployment status in Studio > Endpoints. As long as it shows "Updating", it is still working. |
| `AuthorizationFailure` during deployment | This usually means a storage policy issue. Ask the workshop facilitator for help. It may require adjusting storage account settings. |
| `ResourceNotFound` for model | Make sure Notebook 03 completed successfully and the model is registered. Check **Models** in Studio. |
| Test cell returns error | Check the deployment logs: go to Endpoints > your endpoint > Logs. Common cause: the deployment is still provisioning (not yet ready). |

---

## 9. Part 8: Notebook 05 - Model Monitoring ~20 min

**Open**: `notebooks/track_a_text/05_model_monitoring.ipynb`

### What this notebook does
Sets up automated monitoring for your deployed model. It configures three monitoring signals:
1. **Data drift** — detects when incoming data looks different from training data
2. **Prediction drift** — detects when the model's output distribution changes
3. **Data quality** — catches null values, unexpected formats

It creates a weekly schedule that runs every Monday at 6 AM.

### Before you run

1. Update `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME`
2. **IMPORTANT**: Update `ENDPOINT_NAME` to match the exact name you used in Notebook 04 (with your initials). Look for the `<<<< CHANGE THIS` comment.
3. **IMPORTANT**: Update the `emails` field to your own email address. Look for `<<<< CHANGE THIS TO YOUR EMAIL ADDRESS`.

### Run it
- The training data baseline analysis runs instantly
- The monitoring signals configuration runs instantly
- **Cell 8** (Create schedule): This submits the monitoring schedule to Azure ML

### What to look for in Azure ML Studio
Go to **Monitoring** in the left sidebar. You should see `contoso-lead-monitor` scheduled. The first monitoring run will execute on the next Monday at 6 AM (or you can trigger it manually from Studio).

### What to do if it fails

| Error | Fix |
|-------|-----|
| `ResourceNotFoundError` for endpoint | Your endpoint name in this notebook doesn't match the one from Notebook 04. Make sure `ENDPOINT_NAME` is identical. |
| `ValidationError` | Check that your endpoint has a deployment with traffic routed to it (Notebook 04, cells 5-6 must have completed). |

---

## 10. Part 9: Notebook 06 - ML Pipelines ~20 min

**Open**: `notebooks/track_a_text/06_pipeline_definition.ipynb`

### What this notebook does
Defines a reusable training pipeline using the Azure ML SDK v2. In production, pipelines automate the training workflow — they can be scheduled, triggered by drift alerts, or integrated into CI/CD.

### Before you run
Update the `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, and `WORKSPACE_NAME` values.

### Run it
- **Cell 3**: Registers a custom training environment (builds a Docker container — takes 2-3 minutes the first time)
- **Cell 4**: Defines the training component
- **Cell 8**: Submits the pipeline job. You will get a Studio URL — click it to watch the pipeline run in real time.
- **Cell 9**: Streams the pipeline output (wait for completion, ~5-10 minutes)
- **Cell 11**: Submits two more pipeline runs with different model types

### What to look for in Azure ML Studio
Go to **Jobs** in the left sidebar. Click on `contoso-lead-classifier-pipeline` to see:
- The pipeline DAG (directed acyclic graph) visualization
- Click on individual steps to see their logs and metrics
- Compare multiple pipeline runs

### What to do if it fails

| Error | Fix |
|-------|-----|
| `ResourceNotFoundError` for `cpu-cluster` | You skipped Step 2.6. Go to **Compute** > **Compute clusters** and create `cpu-cluster` as described above. |
| `ImageBuildFailure` | The environment Docker build failed. Check the build logs in Studio (Jobs > your job > Outputs + logs). Usually a network issue — retry by re-running the cell. |
| Pipeline stuck in "Preparing" | The compute cluster is scaling up from 0 nodes. This takes 3-5 minutes on first run. |

---

## 11. Cleanup

**IMPORTANT**: Deployed endpoints cost money while running. Clean up after the workshop.

### Delete the Endpoint

In the last cell of Notebook 04, uncomment and run:

```python
ml_client.online_endpoints.begin_delete(name=ENDPOINT_NAME).result()
print(f"Endpoint '{ENDPOINT_NAME}' deleted.")
```

### Delete the Monitoring Schedule

In Azure ML Studio:
1. Go to **Monitoring** in the left sidebar
2. Click on `contoso-lead-monitor`
3. Click **Delete**

### Delete the Compute Instance (optional)

If you are done with the workshop and won't do Track B:
1. Go to **Compute** > **Compute instances**
2. Select your compute instance
3. Click **Delete**

### Delete All Resources (optional)

To remove everything at once, delete the resource group:
1. Go to [portal.azure.com](https://portal.azure.com)
2. Navigate to **Resource groups**
3. Click on `rg-aml-workshop-{YOUR_INITIALS}`
4. Click **Delete resource group**
5. Type the resource group name to confirm, then click **Delete**

This removes the workspace, storage account, compute, and all associated resources.

---

## 12. Troubleshooting

### Common Issues Across All Notebooks

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Wrong credentials** | `SUBSCRIPTION_ID`, `RESOURCE_GROUP`, or `WORKSPACE_NAME` don't match your Azure resources | Look for `<<<< CHANGE THIS` comments in every notebook. These values must match what you created in Part 1. |
| **Kernel dies** | Notebook kernel crashes mid-execution | Go to **Compute** and restart your compute instance. Then re-open the notebook and re-run from the beginning. |
| **Module not found** | `ModuleNotFoundError: No module named 'azure.ai.ml'` | Run `pip install -r requirements.txt` in a terminal, then restart the kernel. |
| **Stale connection** | `HttpResponseError` or timeout errors | Your Azure token may have expired. Restart the kernel and re-run the connection cell. |
| **File not found** | `FileNotFoundError` for data files or scripts | Your working directory may be wrong. Make sure you cloned the repo correctly and the notebook is in its expected location inside the repo. |
| **Quota exceeded** | `OperationNotAllowed: Quota exceeded` | The shared subscription has hit a resource limit. Try using a smaller VM size (e.g., `Standard_DS2_v2`), or ask the facilitator to request a quota increase. |
| **Endpoint name conflict** | `Endpoint already exists` | Another participant used the same name. Add more characters to `ENDPOINT_NAME` to make it unique. |

### How to Check Notebook 04 Deployment Status

Deployments take the longest (5-10 minutes). If you are unsure whether it is working:

1. Go to **Endpoints** in Azure ML Studio
2. Click on your endpoint name
3. Check the **Provisioning state**:
   - `Updating` — still deploying. Wait.
   - `Succeeded` — ready to test.
   - `Failed` — click on the deployment name, then check **Logs** for details.

### Getting Help

If you are stuck:
1. Check the error message carefully — most issues are credential or naming mismatches
2. Try restarting your compute instance and re-running from the beginning of the notebook
3. Ask a neighbor — they may have hit the same issue
4. Raise your hand for facilitator help
