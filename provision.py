"""Provision Azure ML resources for the MLOps workshop."""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace, AmlCompute, ComputeInstance
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "<YOUR_SUBSCRIPTION_ID>"  # <<<< CHANGE THIS TO YOUR AZURE SUBSCRIPTION ID
RESOURCE_GROUP = "<YOUR_RESOURCE_GROUP>"  # <<<< CHANGE THIS TO YOUR RESOURCE GROUP (e.g., rg-aml-workshop-jd)
WORKSPACE_NAME = "<YOUR_WORKSPACE_NAME>"  # <<<< CHANGE THIS TO YOUR WORKSPACE NAME (e.g., aml-workshop-jd)
LOCATION = "eastus2"

credential = DefaultAzureCredential()

print("=" * 60)
print("Step 1: Creating Azure ML Workspace...")
print("=" * 60)

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
)

ws = Workspace(
    name=WORKSPACE_NAME,
    location=LOCATION,
    display_name="Azure ML MLOps Workshop",
    description="Hands-on MLOps workshop - inspection lead classification and service order repair prediction",
)

workspace = ml_client.workspaces.begin_create(ws).result()
print(f"Workspace created: {workspace.name}")
print(f"  Location: {workspace.location}")
print(f"  Resource group: {workspace.resource_group}")

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

print()
print("=" * 60)
print("Step 2: Creating Compute Cluster (cpu-cluster)...")
print("=" * 60)

cluster = AmlCompute(
    name="cpu-cluster",
    type="amlcompute",
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=300,
)

cluster = ml_client.compute.begin_create_or_update(cluster).result()
print(f"Compute cluster created: {cluster.name}")
print(f"  Size: {cluster.size}")
print(f"  Min/Max instances: {cluster.min_instances}/{cluster.max_instances}")

print()
print("=" * 60)
print("Step 3: Creating Compute Instance (workshop-notebook)...")
print("=" * 60)

ci = ComputeInstance(
    name="workshop-notebook",  # <<<< CHANGE THIS - ADD YOUR INITIALS (e.g., workshop-notebook-jd)
    size="Standard_DS3_v2",
)

ci = ml_client.compute.begin_create_or_update(ci).result()
print(f"Compute instance created: {ci.name}")
print(f"  Size: {ci.size}")
print(f"  State: {ci.state}")

print()
print("=" * 60)
print("ALL RESOURCES PROVISIONED SUCCESSFULLY")
print("=" * 60)
print(f"  Subscription: {SUBSCRIPTION_ID}")
print(f"  Resource Group: {RESOURCE_GROUP}")
print(f"  Workspace: {WORKSPACE_NAME}")
print(f"  Location: {LOCATION}")
print(f"  Compute Cluster: cpu-cluster")
print(f"  Compute Instance: workshop-notebook")
