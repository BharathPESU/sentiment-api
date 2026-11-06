# flows/retrain_flow.py
"""Retrain the model, then promote it only if it passes the quality gate."""
import json
import subprocess
from pathlib import Path

from prefect import flow, get_run_logger, task

ACCURACY_THRESHOLD = 0.85  # choose a value that fits your baseline

@task(retries=1, retry_delay_seconds=30)
def run_pipeline() -> None:
    """Run the DVC pipeline. It reruns only stages whose inputs changed."""
    subprocess.run(["dvc", "repro"], check=True)

@task
def read_accuracy() -> float:
    return json.loads(Path("metrics.json").read_text())["accuracy"]

@flow(name="sentiment-retrain")
def retrain_flow() -> None:
    logger = get_run_logger()
    run_pipeline()
    accuracy = read_accuracy()
    logger.info("Model accuracy: %.4f", accuracy)

    if accuracy < ACCURACY_THRESHOLD:
        raise ValueError(
            f"Accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}. "
            "Not promoting this model."
        )
    logger.info("Quality gate passed")

if __name__ == "__main__":
    retrain_flow()