from prefect import flow, task
import subprocess
import sys

PY = sys.executable  # guarantees we use .venv interpreter when flow is run from .venv


@task
def run_module(module: str):
    subprocess.check_call([PY, "-m", module])


@flow(name="churn_daily_pipeline")
def churn_daily_pipeline():
    # ETL
    run_module("src.etl.download_data")
    run_module("src.etl.init_warehouse")
    run_module("src.etl.load_snapshot")
    run_module("src.etl.build_staging")
    run_module("src.etl.build_features")
    run_module("src.etl.checks")

    # ML + Explainability
    run_module("src.ml.train_xgboost")
    run_module("src.ml.score_xgboost")
    run_module("src.ml.explain.generate_shap")

    # Monitoring
    run_module("src.ml.monitoring.compute_monitoring_metrics")


if __name__ == "__main__":
    churn_daily_pipeline()