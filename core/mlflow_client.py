import os
from pathlib import Path

import dagshub
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def init_mlflow(experiment_name: str) -> None:
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")
    repo = os.getenv("DAGSHUB_REPO")
    dagshub.init(repo_owner=f"{username}", repo_name=f"{repo}", mlflow=True)

    if not username or not token or not repo:
        return
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()


def log_params(params: dict) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: dict) -> None:
    mlflow.log_metrics(metrics)


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)


def register_model(artifact_path: str, name: str, tags: dict) -> None:
    artifact_name = Path(artifact_path).name
    mlflow.log_artifact(artifact_path)
    run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    run = client.get_run(run_id)
    source = f"{run.info.artifact_uri}/{artifact_name}"

    try:
        client.create_registered_model(name, tags=tags)
    except mlflow.exceptions.MlflowException:
        pass  # model already exists

    client.create_model_version(name=name, source=source, run_id=run_id, tags=tags)
