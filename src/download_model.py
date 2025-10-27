# download_model.py
import mlflow.artifacts
import os
import sys

def get_model(model_uri, destination_path):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Attempting to download artifacts for {model_uri} to {destination_path}...")

    try:
        # create the destination dir if it doesn't exist
        os.makedirs(destination_path, exist_ok=True)
        print(f"Ensured directory exists: {destination_path}")

        # Download the artifacts
        actual_download_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=destination_path,
            tracking_uri=tracking_uri # for clarity
        )
        print(f"Successfully downloaded artifacts to: {actual_download_path}")

        # check contents
        if os.path.exists(destination_path) and os.path.isdir(destination_path):
            print(f"Contents of {destination_path}:")
            for item in os.listdir(destination_path):
                print(f"- {item}")
            if not os.listdir(destination_path):
                print(f"Warning: {destination_path} is empty after download attempt.")
                # Optionally fail the job if the directory is empty
        else:
            print(f"Error: Destination path {destination_path} does not exist or is not a directory after download attempt.")
            sys.exit(1) # Exit with error code if download verification fails

    except Exception as e:
        print(f"Error downloading artifacts: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1) # Exit with error code if download fails

    print("Model download script finished successfully.")

if __name__ == "__main__":

    # NON - LAG MODELS 
    model_uri_svr_nolag = "models:/model_linearSVR_scaler_batches_nolag/latest"
    destination_path_svr_nolag = "src/serve_model_svr_nolag"
    
    model_uri_xgb_nolag = "models:/model_xgb_model_scaler_batches_nolag/latest"
    destination_path_xgb_nolag = "src/serve_model_xgb_nolag"

    model_uri_randomForest_nolag = "models:/model_randomforest_model_scaler_batches_nolag/latest"
    destination_path_randomForest_nolag = "src/serve_model_randomForest_nolag"
    
    get_model(model_uri_svr_nolag, destination_path_svr_nolag)
    get_model(model_uri_xgb_nolag, destination_path_xgb_nolag)
    get_model(model_uri_randomForest_nolag, destination_path_randomForest_nolag)

    # LAG - MODELS
    model_uri_svr_lag = "models:/model_linearSVR_scaler_batches/latest"
    destination_path_svr_lag = "src/serve_model_svr_lag"
    
    model_uri_xgb_lag = "models:/model_xgb_model_scaler_batches/latest"
    destination_path_xgb_lag = "src/serve_model_xgb_lag"

    model_uri_randomForest_lag = "models:/model_randomforest_model_scaler_batches_lag/latest"
    destination_path_randomForest_lag = "src/serve_model_randomForest_lag"
    
    get_model(model_uri_svr_lag, destination_path_svr_lag)
    get_model(model_uri_xgb_lag, destination_path_xgb_lag)
    get_model(model_uri_randomForest_lag, destination_path_randomForest_lag)