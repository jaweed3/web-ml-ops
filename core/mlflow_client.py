import os
import mlflow
from dotenv import load_dotenv
load_dotenv()

def init_mlflow(experiment_name: str) -> None:
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")
    repo = os.getenv("DAGSHUB_REPO")
    
    if not username or not token or not repo:
        return

    uri = f"https://dagshub.com/{username}/{repo}.mlflow"

    mlflow.set_tracking_uri(uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
        

def log_params(params: dict) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: dict) -> None:
    mlflow.log_metrics(metrics)


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)


def register_model(artifact_path: str, name: str, tags: dict) -> None:
    mlflow.log_artifact(artifact_path)
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mlflow.register_model(model_uri=model_uri, name=name, tags=tags)
