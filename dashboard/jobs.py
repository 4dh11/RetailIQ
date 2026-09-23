"""
Background retraining job, run by an RQ worker (not by the Flask process).

Runs the three notebooks headlessly, in dependency order, then stamps
data/last_trained.json. Each notebook is executed in place with nbconvert,
so no refactor of the notebooks themselves is needed.

Left out on purpose: scoring the 5,878 customers with the already-fitted
churn model. That's milliseconds of inference, not retraining, so it stays
on the request path (or inside churn_model.ipynb's own scoring cell) instead
of going through the queue.
"""

import json
import os
import subprocess
import time

from rq import get_current_job

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(DASHBOARD_DIR)  # repo root

NOTEBOOK_DIR = os.path.join(BASE, "notebook")
MODEL_DIR = os.path.join(BASE, "models")
DATA_DIR = os.path.join(BASE, "data")

LAST_TRAINED_PATH = os.path.join(DATA_DIR, "last_trained.json")

# (status label shown in the UI, notebook path, rough duration for the timeout)
STEPS = [
    ("Rebuilding data pipeline", os.path.join(NOTEBOOK_DIR, "data_exploration.ipynb")),
    ("Refitting churn model", os.path.join(MODEL_DIR, "churn_model.ipynb")),
    ("Refitting sales forecast", os.path.join(MODEL_DIR, "sales_forecasting.ipynb")),
]

NOTEBOOK_TIMEOUT_SECONDS = 1800  # per-cell timeout inside nbconvert


def _set_step(label):
    """Publish progress onto the RQ job so /retrain/status can show it while running."""
    job = get_current_job()
    if job is not None:
        job.meta["step"] = label
        job.save_meta()


def _run_notebook(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Notebook not found: {path}")

    result = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={NOTEBOOK_TIMEOUT_SECONDS}",
            path,
        ],
        cwd=os.path.dirname(path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # nbconvert's own traceback is the most useful part; surface the tail of it.
        tail = "\n".join(result.stderr.strip().splitlines()[-25:])
        raise RuntimeError(f"{os.path.basename(path)} failed:\n{tail}")


def retrain_pipeline():
    """
    Entry point enqueued by the Flask app. Raises on the first failing notebook,
    which RQ turns into a 'failed' job status with job.exc_info set.
    """
    for label, path in STEPS:
        _set_step(label)
        _run_notebook(path)

    _set_step("Finishing up")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LAST_TRAINED_PATH, "w") as f:
        json.dump({"timestamp": time.time()}, f)

    return {"status": "ok"}
