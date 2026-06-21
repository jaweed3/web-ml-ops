from pathlib import Path

from app.monitoring.drift import _BASELINE_FILE, compute_baseline
from app.utils.common import write_json
from core.logger import get_logger

log = get_logger("scripts.compute_baseline")

if __name__ == "__main__":
    train_dir = Path("train_data")
    if not train_dir.exists():
        log.error("train_data_not_found", hint="run `dvc pull` first")
        raise SystemExit(1)

    baseline = compute_baseline(train_dir)
    write_json(baseline, _BASELINE_FILE)
    log.info("baseline_saved", path=str(_BASELINE_FILE), keys=str(list(baseline.keys())))
