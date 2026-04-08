import os
import mlflow


def init_mlflow(experiment_name: str) -> None:
    uri = (
        f"https://dagshub.com/"
        f"{os.environ['DAGSHUB_USERNAME']}/"
        f"{os.environ['DAGSHUB_REPO']}.mlflow"
    )
    mlflow.set_tracking_uri(uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
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
