import time

from core.logger import get_logger
from core.config import load_config
from core.mlflow_client import init_mlflow, log_params, log_metrics, log_artifact

log = get_logger("stage2_train")


def train(cfg) -> str:
    from ultralytics import YOLO
    init_mlflow(experiment_name="rescuevision-yolov8n")

    log.info("training_start", extra={
        "device":cfg.train.device, 
        "model":cfg.model.name, 
        "epochs":cfg.train.epochs,
    })
    
    log_params({
        "model":     cfg.model.name,
        "epochs":    cfg.train.epochs,
        "imgsz":     cfg.train.imgsz,
        "batch":     cfg.train.batch,
        "lr0":       cfg.train.lr0,
        "optimizer": cfg.train.optimizer,
        "device":    cfg.train.device,
    })

    model = YOLO(cfg.model.name)
    t0 = time.time()

    results = model.train(
        data=cfg.data.yaml,
        epochs=cfg.train.epochs,
        imgsz=cfg.train.imgsz,
        batch=cfg.train.batch,
        lr0=cfg.train.lr0,
        optimizer=cfg.train.optimizer,
        device=cfg.train.device,
        project="runs",
        name="train",
        exist_ok=True,
    )

    training_time = round(time.time() - t0, 2)
    metrics = {
        "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
        "mAP50_95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "precision": results.results_dict.get("metrics/precision(B)", 0),
        "recall": results.results_dict.get("metrics/recall(B)", 0),
        "train_time_seconds": training_time,
    }
    log_metrics(metrics)
    log.info("training_complete", **metrics)

    checkpoint_path = "runs/train/weights/best.pt"
    log_artifact(checkpoint_path)
    log.info("checkpoint_logged", path=checkpoint_path)

    return checkpoint_path


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
