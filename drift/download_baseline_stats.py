import mlflow
import os
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000") 

REGISTERED_MODEL_NAME = "model_linearSVR_scaler"
MLFLOW_TRACKING_URI = "http://localhost:5000"


artifact_to_download = "training_feature_stats.json" 
local_dir = "drift/"
os.makedirs(local_dir, exist_ok=True) 


print(f"Connecting to MLflow at: {MLFLOW_TRACKING_URI}")
client = MlflowClient()

print(f"Searching for the latest version of registered model: '{REGISTERED_MODEL_NAME}'")
latest_versions = client.get_latest_versions(name=REGISTERED_MODEL_NAME, stages=None)

if not latest_versions:
        print(f"ERROR: No versions found for registered model '{REGISTERED_MODEL_NAME}'.")
        exit()


latest_model_version = latest_versions[0]
run_id = latest_model_version.run_id
version_number = latest_model_version.version

print(f"Found latest version: {version_number}, associated with run_id: '{run_id}'")


print(f"Attempting to download artifact: '{artifact_to_download}' from run_id: '{run_id}'")

client = MlflowClient()
local_path_file = client.download_artifacts(
    run_id=run_id,
    path=artifact_to_download, # Use the variable
    dst_path=local_dir         # Specify destination directory
)


expected_local_file_path = os.path.join(local_dir, artifact_to_download)

print(f"Artifact '{artifact_to_download}' downloaded to: '{local_path_file}'") 
if os.path.exists(expected_local_file_path):
        print(f"Verified: File exists at '{expected_local_file_path}'")
else:
        print(f"Warning: Expected file not found at '{expected_local_file_path}'. Check download path logic.")
