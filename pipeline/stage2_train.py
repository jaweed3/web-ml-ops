import os
import time
from pathlib import Path

from core.config import is_subset_mode, load_config
from core.logger import get_logger
from core.mlflow_client import init_mlflow, log_artifact, log_metrics, log_params

log = get_logger("stage2_train")


def train(cfg) -> str:
    from ultralytics import YOLO

    init_mlflow(experiment_name="rescuevision-yolov8n")

    log.info(
        "training_start",
        extra={
            "device": cfg.train.device,
            "model": cfg.model.name,
            "epochs": cfg.train.epochs,
        },
    )

    log_params(
        {
            "model": cfg.model.name,
            "epochs": cfg.train.epochs,
            "imgsz": cfg.train.imgsz,
            "batch": cfg.train.batch,
            "lr0": cfg.train.lr0,
            "optimizer": cfg.train.optimizer,
            "device": cfg.train.device,
        }
    )

    model = YOLO(cfg.model.name)
    t0 = time.time()

    data_yaml = cfg.data.subset_yaml if is_subset_mode() else cfg.data.yaml
    device = os.getenv("TRAIN_DEVICE", cfg.train.device)
    results = model.train(
        data=data_yaml,
        epochs=cfg.train.epochs,
        imgsz=cfg.train.imgsz,
        batch=cfg.train.batch,
        lr0=cfg.train.lr0,
        optimizer=cfg.train.optimizer,
        device=device,
        project=str(Path.cwd() / "runs"),
        name="train",
        exist_ok=True,
    )

    assert results is not None
    training_time = round(time.time() - t0, 2)
    metrics = {
        "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
        "mAP50_95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "precision": results.results_dict.get("metrics/precision(B)", 0),
        "recall": results.results_dict.get("metrics/recall(B)", 0),
        "train_time_seconds": training_time,
    }
    log_metrics(metrics)
    log.info("training_complete", extra=metrics)

    weights_dir = results.save_dir / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    checkpoint_path = str(best if best.exists() else last)

    # Write path so downstream stages can find it without guessing
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/checkpoint_path.txt").write_text(checkpoint_path)

    log_artifact(checkpoint_path)
    log.info("checkpoint_logged", extra={"path": checkpoint_path})

    return checkpoint_path


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
