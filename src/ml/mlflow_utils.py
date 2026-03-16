from pathlib import Path
import mlflow

def setup_mlflow():
    # Local tracking folder in your repo
    mlruns_path = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment("churn_prediction")