import subprocess
import sys

PY = sys.executable

def run_module(module: str):
    subprocess.check_call([PY, "-m", module])

def main():
    run_module("src.etl.download_data")
    run_module("src.etl.init_warehouse")
    run_module("src.etl.load_snapshot")
    run_module("src.etl.build_staging")
    run_module("src.etl.build_features")
    run_module("src.etl.checks")

    run_module("src.ml.train_xgboost")
    run_module("src.ml.score_xgboost")
    run_module("src.ml.explain.generate_shap")
    run_module("src.ml.monitoring.compute_monitoring_metrics")

if __name__ == "__main__":
    main()