# Azure ML MLOps Workshop

Hands-on workshop implementing two ML use cases on Azure ML, covering the full MLOps lifecycle with both **text** and **tabular** data.

## Quick Start

**New here?** Follow the step-by-step guides in the [`workshop_instructions/`](workshop_instructions/) folder:

| Guide | Duration | Description |
|-------|----------|-------------|
| [**Track A: Text Classification**](workshop_instructions/track_a_text_guide.md) | ~3 hours | Classify inspection comments as sales leads. **Start here.** |
| [**Track B: Tabular Classification**](workshop_instructions/track_b_tabular_guide.md) | ~2 hours | Predict service order repair type. Self-paced take-home. |

The guides walk you through everything from creating Azure resources to deploying a live endpoint — no prior Azure ML experience required.

## The Use Cases

| Track | Dataset | Rows | ML Problem | Target |
|-------|---------|------|-----------|--------|
| **Track A** (Text) | `inspections_dataset.csv` | 10,500 | Classify inspection comments as sales leads | `is_lead_opportunity` |
| **Track B** (Tabular) | `service_orders_dataset.csv` | 425,745 | Predict service order repair type | `RepairType` |

**Why two tracks?** The same MLOps infrastructure (versioning, tracking, deployment, monitoring, pipelines) works identically regardless of data type. Participants see this firsthand by building two complete ML systems.

## Generating the Sample Data

The datasets are not included in the repository. To generate synthetic sample data, run:

```bash
pip install faker openpyxl pandas numpy
python generate_sample_data.py
```

This creates `data/inspections_dataset.csv` and `data/service_orders_dataset.csv` with realistic synthetic data matching the expected schemas.

## Workshop Format

- **Track A** is completed during the 3-hour live workshop
- **Track B** is a self-paced take-home exercise that reinforces the same concepts on tabular data
- All participants share one Azure subscription — each person creates their own resource group and workspace (instructions in the guides)
- Every notebook has `<<<< CHANGE THIS` comments marking values you need to update

## Workshop Agenda (Track A)

| Time | Topic | Notebook |
|------|-------|----------|
| 0:00 - 0:20 | Azure setup (portal) | Create resource group, workspace, compute |
| 0:20 - 0:30 | Clone repo & install deps | `notebooks/00_setup_and_config.ipynb` |
| 0:30 - 0:45 | Data version control | `notebooks/track_a_text/01_data_versioning.ipynb` |
| 0:45 - 1:15 | Experiment tracking | `notebooks/track_a_text/02_experiment_tracking.ipynb` |
| 1:15 - 1:30 | Model registration | `notebooks/track_a_text/03_model_registration.ipynb` |
| 1:30 - 2:00 | Model deployment & serving | `notebooks/track_a_text/04_model_deployment.ipynb` |
| 2:00 - 2:20 | Model monitoring | `notebooks/track_a_text/05_model_monitoring.ipynb` |
| 2:20 - 2:45 | Pipeline orchestration | `notebooks/track_a_text/06_pipeline_definition.ipynb` |
| 2:45 - 3:00 | Cleanup & wrap-up | Delete endpoints to avoid costs |

## Prerequisites

- **Azure subscription** with Contributor access (shared subscription)
- **Web browser** (Edge or Chrome)
- No local software installation required — everything runs on Azure ML compute instances

All Azure resources (resource group, workspace, compute) are created from scratch during the workshop. See the [Track A guide](workshop_instructions/track_a_text_guide.md) for detailed setup instructions.

## Project Structure

```
AML_MLOps_101/
├── README.md
├── requirements.txt                              # Python dependencies
├── provision.py                                  # Optional: script to provision Azure resources
├── generate_sample_data.py                       # Script to generate synthetic datasets
│
├── workshop_instructions/                        # <<<< START HERE
│   ├── track_a_text_guide.md                     #     Step-by-step guide for Track A
│   └── track_b_tabular_guide.md                  #     Step-by-step guide for Track B
│
├── data/
│   ├── inspections_dataset.csv                   # Track A dataset (generated)
│   └── service_orders_dataset.csv                # Track B dataset (generated)
│
├── notebooks/
│   ├── 00_setup_and_config.ipynb                 # Shared setup (both tracks)
│   │
│   ├── track_a_text/                             # ── Track A: Text Classification ──
│   │   ├── 01_data_versioning.ipynb
│   │   ├── 02_experiment_tracking.ipynb
│   │   ├── 03_model_registration.ipynb
│   │   ├── 04_model_deployment.ipynb
│   │   ├── 05_model_monitoring.ipynb
│   │   └── 06_pipeline_definition.ipynb
│   │
│   └── track_b_tabular/                          # ── Track B: Tabular Classification ──
│       ├── 01b_data_versioning_tabular.ipynb
│       ├── 02b_experiment_tracking_tabular.ipynb
│       ├── 03b_model_registration_tabular.ipynb
│       ├── 04b_model_deployment_tabular.ipynb
│       ├── 05b_model_monitoring_tabular.ipynb
│       └── 06b_pipeline_definition_tabular.ipynb
│
├── src/
│   ├── track_a_text/                             # ── Track A source code ──
│   │   ├── preprocess.py                         #     Text cleaning & TF-IDF features
│   │   ├── train.py                              #     Text classifier training script
│   │   └── score.py                              #     Text classifier scoring script
│   │
│   └── track_b_tabular/                          # ── Track B source code ──
│       ├── preprocess_os.py                      #     Tabular cleaning & feature engineering
│       ├── train_os.py                           #     Tabular classifier training script
│       └── score_os.py                           #     Tabular classifier scoring script
│
├── config/
│   ├── track_a_text/                             # ── Track A YAML configs ──
│   │   ├── deployment.yml
│   │   ├── monitoring.yml
│   │   └── pipeline.yml
│   │
│   └── track_b_tabular/                          # ── Track B YAML configs ──
│       ├── deployment_os.yml
│       ├── monitoring_os.yml
│       └── pipeline_os.yml
│
└── environment/
    ├── conda.yml                                 # Shared training environment
    ├── track_a_text/
    │   └── deployment_env.yml                    #     Text scoring environment
    └── track_b_tabular/
        └── deployment_env_os.yml                 #     Tabular scoring environment
```

## MLOps Topics Covered

Each topic is demonstrated on **both** text and tabular data:

| Topic | Azure ML Feature | Track A | Track B |
|-------|-----------------|---------|---------|
| **Data Version Control** | Data Assets | `classified-inspections` v1/v2 | `service-orders` v1/v2 |
| **Experiment Tracking** | MLflow integration | `contoso-lead-classifier` experiment | `contoso-repair-classifier` experiment |
| **Model Tracking** | Model Registry | `contoso-lead-classifier` model | `contoso-repair-classifier` model |
| **Model Serving** | Managed Online Endpoints | Text endpoint (inspection comments) | Tabular endpoint (structured JSON) |
| **Model Monitoring** | Model Monitor | TF-IDF drift detection | Feature distribution drift |
| **Pipeline Orchestration** | SDK v2 Pipelines | Text training pipeline | Tabular training pipeline |

## Important: Update Hardcoded Values

Every notebook contains hardcoded Azure resource values that **you must update** before running. Look for comments marked with `<<<< CHANGE THIS`:

```python
SUBSCRIPTION_ID = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  # <<<< CHANGE THIS TO YOUR AZURE SUBSCRIPTION ID
RESOURCE_GROUP = "rg-aml-workshop-jd"  # <<<< CHANGE THIS TO YOUR RESOURCE GROUP
WORKSPACE_NAME = "aml-workshop-jd"  # <<<< CHANGE THIS TO YOUR WORKSPACE NAME
```

Additionally, in deployment notebooks (04, 04b), endpoint names must be unique — append your initials:

```python
ENDPOINT_NAME = "contoso-lead-classifier-jd"  # <<<< ADD YOUR INITIALS
```

## Cleanup

**Deployed endpoints cost money while running.** Clean up after the workshop:

1. **Delete endpoints** — uncomment and run the cleanup cell at the bottom of notebooks 04/04b
2. **Delete monitoring schedules** — go to Azure ML Studio > Monitoring > Delete
3. **Delete all resources** — delete your resource group from the Azure Portal to remove everything at once

See the [Cleanup section](workshop_instructions/track_a_text_guide.md#11-cleanup) in the Track A guide for detailed steps.
