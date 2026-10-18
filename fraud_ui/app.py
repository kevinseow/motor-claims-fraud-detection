"""
Fraud Detection UI — Flask backend with Decoupled Cloud Tasks Queue.

Architecture:
  1. POST /api/upload
     - Ingestion: Uploads claim pack (PDFs + photos) to Google Cloud Storage
     - Decoupling: Enqueues processing task into Cloud Tasks queue (rate-limited dispatch)
     - Immediate return: Client receives {run_id, case_id, status: 'queued'}
  
  2. POST /api/tasks/process-claim
     - Processing worker invoked by Cloud Tasks (respecting max_dispatches_per_second)
     - Triggers Cloud Run Job execution (fraud-detection-job)
     - Retries automatically with exponential backoff on transient errors (max 5 attempts)

  3. GET /api/status
     - Returns live status: 'queued' -> 'running' (with execution details) -> 'succeeded' / 'failed'

  4. GET /api/results
     - Returns FRAUD_DETECTION_RESULTS.md from Cloud Storage once complete

Environment variables:
  AUTH_ENABLED        Toggle access key auth: "true" or "false" (default: false for public access)
  UI_API_KEY          Bearer token clients & Cloud Tasks send in X-API-Key header (when AUTH_ENABLED=true)
  GCS_BUCKET          Cloud Storage bucket name for inputs/outputs
  GCP_PROJECT         Google Cloud project ID
  GCP_REGION          Cloud Tasks & Cloud Run region (e.g. asia-southeast1)
  CLOUD_TASKS_QUEUE   Name of Cloud Tasks queue (default: fraud-claims-queue)
  SERVICE_URL         Public URL of this Cloud Run service for task callbacks
  FRAUD_RUN_JOB       Name of the Cloud Run Job to trigger (default: fraud-detection-job)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from google.cloud import run_v2, storage, tasks_v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_ui")

app = Flask(__name__, static_folder="static")

# ── config ────────────────────────────────────────────────────────────────────

AUTH_ENABLED      = os.environ.get("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
UI_API_KEY        = os.environ.get("UI_API_KEY", "")
GCS_BUCKET        = os.environ.get("GCS_BUCKET", "fraud-detection-511308-fraud-data")
GCP_PROJECT       = os.environ.get("GCP_PROJECT", "fraud-detection-511308")
GCP_REGION        = os.environ.get("GCP_REGION", "asia-southeast1")
JOB_NAME          = os.environ.get("FRAUD_RUN_JOB", "fraud-detection-job")
CLOUD_TASKS_QUEUE = os.environ.get("CLOUD_TASKS_QUEUE", "fraud-claims-queue")
SERVICE_URL       = os.environ.get("SERVICE_URL", "https://fraud-ui-1082243885044.asia-southeast1.run.app")

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

# ── auth ──────────────────────────────────────────────────────────────────────

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_ENABLED:
            return f(*args, **kwargs)
        key = request.headers.get("X-API-Key", "")
        if not UI_API_KEY or key != UI_API_KEY:
            logger.warning("Unauthorized request attempt")
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ── helpers ───────────────────────────────────────────────────────────────────

def _gcs_client() -> storage.Client:
    return storage.Client(project=GCP_PROJECT)


def _run_client() -> run_v2.JobsClient:
    return run_v2.JobsClient()


def _tasks_client() -> tasks_v2.CloudTasksClient:
    return tasks_v2.CloudTasksClient()


def _save_status(run_id: str, status_data: dict) -> None:
    try:
        client = _gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"outputs/{run_id}/status.json")
        blob.upload_from_string(
            json.dumps(status_data, indent=2),
            content_type="application/json",
        )
    except Exception as e:
        logger.error(f"Failed to save status for {run_id}: {e}")


def _load_status(run_id: str) -> dict | None:
    try:
        client = _gcs_client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"outputs/{run_id}/status.json")
        if blob.exists():
            return json.loads(blob.download_as_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load status for {run_id}: {e}")
    return None


def _trigger_job(run_id: str, case_id: str) -> str:
    jobs_client = _run_client()
    job_resource = f"projects/{GCP_PROJECT}/locations/{GCP_REGION}/jobs/{JOB_NAME}"

    execution_request = run_v2.RunJobRequest(
        name=job_resource,
        overrides=run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    env=[
                        run_v2.EnvVar(name="GCS_BUCKET",   value=GCS_BUCKET),
                        run_v2.EnvVar(name="GCS_RUN_ID",   value=run_id),
                        run_v2.EnvVar(name="GCS_CASE_ID",  value=case_id),
                        run_v2.EnvVar(name="FRAUD_CASE_FILTER", value=case_id),
                        run_v2.EnvVar(name="FRAUD_RPM",    value=os.environ.get("FRAUD_RPM", "5")),
                        run_v2.EnvVar(name="FRAUD_VISION_CHAIN", value=os.environ.get("FRAUD_VISION_CHAIN", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash")),
                        run_v2.EnvVar(name="FRAUD_TEXT_CHAIN",   value=os.environ.get("FRAUD_TEXT_CHAIN", "gemini-3.7-flash,gemini-3.5-flash-lite,gemini-2.5-flash")),
                    ],
                    args=[],
                )
            ]
        ),
    )
    logger.info(f"Triggering Cloud Run Job {JOB_NAME} for case {case_id} (run_id={run_id})...")
    op = jobs_client.run_job(request=execution_request)
    return op.metadata.name


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/config")
def get_config():
    return jsonify({
        "auth_enabled": AUTH_ENABLED,
    })


@app.route("/api/upload", methods=["POST"])
@require_api_key
def upload():
    """
    Decoupled Ingestion:
    1. Accepts claim pack files and uploads to Cloud Storage:
       inputs/<run_id>/cases/<case_id>/             <- PDFs
       inputs/<run_id>/cases/<case_id>/evidence/    <- Images
    2. Writes initial state ('queued') to GCS
    3. Enqueues a task in Cloud Tasks for rate-controlled execution
    4. Immediately returns 202 Accepted with run_id
    """
    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    case_id = (request.form.get("case_id") or "CASE-" + uuid.uuid4().hex[:6].upper()).upper()
    run_id  = uuid.uuid4().hex[:8]

    # Validate extensions
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"unsupported file type: {f.filename}"}), 400

    # 1. Upload files to GCS
    client = _gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    uploaded = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            blob_path = f"inputs/{run_id}/cases/{case_id}/evidence/{f.filename}"
        else:
            blob_path = f"inputs/{run_id}/cases/{case_id}/{f.filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_file(f.stream, content_type=f.content_type)
        uploaded.append(blob_path)

    # 2. Record initial status in GCS
    now_iso = datetime.now(timezone.utc).isoformat()
    status_data = {
        "run_id":         run_id,
        "case_id":        case_id,
        "state":          "queued",
        "message":        "Enqueued in Cloud Tasks rate-limiting queue",
        "created_at":     now_iso,
        "execution_name": None,
        "files_uploaded": len(uploaded),
    }
    _save_status(run_id, status_data)

    # 3. Enqueue task in Cloud Tasks
    task_created = False
    task_name = None
    try:
        tasks_client = _tasks_client()
        queue_path = tasks_client.queue_path(GCP_PROJECT, GCP_REGION, CLOUD_TASKS_QUEUE)

        # Target worker endpoint on this Cloud Run service
        base_url = (SERVICE_URL or request.host_url).rstrip("/")
        worker_url = f"{base_url}/api/tasks/process-claim"

        payload = {"run_id": run_id, "case_id": case_id}
        task_headers = {"Content-Type": "application/json"}
        if AUTH_ENABLED and UI_API_KEY:
            task_headers["X-API-Key"] = UI_API_KEY

        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": worker_url,
                "headers": task_headers,
                "body": json.dumps(payload).encode(),
            }
        }
        created_task = tasks_client.create_task(request={"parent": queue_path, "task": task})
        task_name = created_task.name
        task_created = True
        logger.info(f"Enqueued Cloud Task: {task_name} for case {case_id}")
    except Exception as e:
        logger.warning(f"Could not enqueue to Cloud Tasks ({e}). Falling back to direct trigger.")
        try:
            execution_name = _trigger_job(run_id, case_id)
            status_data["state"] = "running"
            status_data["execution_name"] = execution_name
            status_data["started_at"] = datetime.now(timezone.utc).isoformat()
            status_data["message"] = f"Job triggered directly: {execution_name.split('/')[-1]}"
            _save_status(run_id, status_data)
            return jsonify({
                "run_id":         run_id,
                "case_id":        case_id,
                "execution_name": execution_name,
                "state":          "running",
                "files_uploaded": uploaded,
            })
        except Exception as direct_err:
            logger.error(f"Fallback trigger failed: {direct_err}")
            return jsonify({"error": f"Failed to start analysis: {direct_err}"}), 500

    return jsonify({
        "run_id":         run_id,
        "case_id":        case_id,
        "task_name":      task_name,
        "state":          "queued",
        "message":        "Claim pack uploaded. Waiting in Cloud Tasks queue for rate-limited dispatch.",
        "files_uploaded": uploaded,
    }), 202


@app.route("/api/tasks/process-claim", methods=["POST"])
@require_api_key
def process_claim_task():
    """
    Cloud Tasks Worker Endpoint:
    Dispatched by Cloud Tasks at a controlled rate (e.g. 10/s, max 20 concurrent).
    Automatically retried with exponential backoff on HTTP 5xx responses.
    """
    payload = request.get_json(silent=True) or {}
    run_id  = payload.get("run_id")
    case_id = payload.get("case_id")

    if not run_id or not case_id:
        logger.error(f"Missing run_id or case_id in task payload: {payload}")
        return jsonify({"error": "invalid payload"}), 400

    logger.info(f"Processing task from Cloud Tasks: run_id={run_id}, case_id={case_id}")

    # Load current status
    status_data = _load_status(run_id) or {
        "run_id": run_id,
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Trigger the Cloud Run Job
        execution_name = _trigger_job(run_id, case_id)

        # Update status to running with the execution resource name
        status_data["state"] = "running"
        status_data["execution_name"] = execution_name
        status_data["started_at"] = datetime.now(timezone.utc).isoformat()
        status_data["message"] = f"Execution active: {execution_name.split('/')[-1]}"
        _save_status(run_id, status_data)

        logger.info(f"Successfully dispatched run_id={run_id} as execution {execution_name}")
        return jsonify({
            "ok": True,
            "run_id": run_id,
            "case_id": case_id,
            "execution_name": execution_name,
        }), 200

    except Exception as e:
        logger.error(f"Error triggering job for run_id={run_id}: {e}")
        status_data["last_error"] = str(e)
        status_data["retrying"] = True
        _save_status(run_id, status_data)
        # Returning 500 triggers Cloud Tasks automatic retry with backoff!
        return jsonify({"error": f"job dispatch failed, retrying: {e}"}), 500


@app.route("/api/status")
@require_api_key
def status():
    """
    Query params:
      run_id          — the run_id from /api/upload (preferred)
      execution_name  — execution resource name (fallback)
    """
    run_id = request.args.get("run_id", "")
    execution_name = request.args.get("execution_name", "")

    status_data = None
    if run_id:
        status_data = _load_status(run_id)

    # If status is still queued, return immediately
    if status_data and status_data.get("state") == "queued":
        return jsonify({
            "run_id":    run_id,
            "case_id":   status_data.get("case_id"),
            "state":     "queued",
            "message":   status_data.get("message", "Waiting in rate-limiting queue..."),
        })

    # Resolve execution name if available from status.json
    if not execution_name and status_data:
        execution_name = status_data.get("execution_name") or ""

    if not execution_name:
        # Check if output report already exists directly in GCS
        if run_id:
            client = _gcs_client()
            bucket = client.bucket(GCS_BUCKET)
            if bucket.blob(f"outputs/{run_id}/FRAUD_DETECTION_RESULTS.md").exists():
                return jsonify({"run_id": run_id, "state": "succeeded"})
        return jsonify({"state": "queued", "message": "Awaiting execution dispatch..."})

    # Query Cloud Run Executions API
    executions_client = run_v2.ExecutionsClient()
    try:
        execution = executions_client.get_execution(name=execution_name)
    except Exception as e:
        logger.warning(f"Could not get execution {execution_name}: {e}")
        return jsonify({"state": "running", "execution_name": execution_name})

    state = "running"
    if execution.succeeded_count and execution.succeeded_count > 0:
        state = "succeeded"
    elif execution.failed_count and execution.failed_count > 0:
        state = "failed"
    elif execution.cancelled_count and execution.cancelled_count > 0:
        state = "cancelled"

    return jsonify({
        "run_id":          run_id,
        "execution_name":  execution_name,
        "state":           state,
        "succeeded":       execution.succeeded_count,
        "failed":          execution.failed_count,
        "start_time":      execution.start_time.isoformat() if execution.start_time else None,
        "completion_time": execution.completion_time.isoformat() if execution.completion_time else None,
    })


@app.route("/api/results")
@require_api_key
def results():
    """
    Query params:
      run_id  — from the /upload response
    Reads FRAUD_DETECTION_RESULTS.md from GCS outputs/<run_id>/
    """
    run_id = request.args.get("run_id", "")
    if not run_id:
        return jsonify({"error": "run_id is required"}), 400

    client = _gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"outputs/{run_id}/FRAUD_DETECTION_RESULTS.md")
    if not blob.exists():
        return jsonify({"error": "results not yet available"}), 404

    return jsonify({
        "run_id":  run_id,
        "report":  blob.download_as_text(encoding="utf-8"),
    })


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
