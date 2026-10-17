"""
Fraud Detection UI — Flask backend.

Serves the upload UI and orchestrates the Cloud Run Job pipeline:
  POST /api/upload   — accepts multipart claim files, writes to GCS, triggers job
  GET  /api/status   — polls the Cloud Run Jobs execution state
  GET  /api/results  — returns the markdown report from GCS once the job is done

Environment variables (all required at runtime):
  UI_API_KEY          simple bearer token clients must send in X-API-Key header
  GCS_BUCKET          Cloud Storage bucket name for inputs/outputs
  GCP_PROJECT         Google Cloud project ID
  GCP_REGION          Cloud Run region (e.g. asia-southeast1)
  CLOUD_RUN_JOB       name of the Cloud Run Job to trigger
"""

from __future__ import annotations

import json
import os
import uuid
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from google.cloud import run_v2, storage

app = Flask(__name__, static_folder="static")

# ── config ────────────────────────────────────────────────────────────────────

UI_API_KEY   = os.environ["UI_API_KEY"]
GCS_BUCKET   = os.environ["GCS_BUCKET"]
GCP_PROJECT  = os.environ["GCP_PROJECT"]
GCP_REGION   = os.environ["GCP_REGION"]
JOB_NAME     = os.environ.get("FRAUD_RUN_JOB", "fraud-detection-job")

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

# ── auth ──────────────────────────────────────────────────────────────────────

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        if key != UI_API_KEY:
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ── helpers ───────────────────────────────────────────────────────────────────

def _gcs_client() -> storage.Client:
    return storage.Client(project=GCP_PROJECT)


def _run_client() -> run_v2.JobsClient:
    return run_v2.JobsClient()


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/upload", methods=["POST"])
@require_api_key
def upload():
    """
    Accepts multipart form fields:
      case_id   — the CASE-NNN reference (optional; auto-generated if absent)
      files[]   — one or more PDF / JPEG files for the claim pack

    Files are written into GCS under:
      inputs/<run_id>/cases/<case_id>/             ← PDFs go here
      inputs/<run_id>/cases/<case_id>/evidence/    ← JPEGs go here

    Then a Cloud Run Job execution is triggered with:
      --args ["--case", case_id, "--score"]
    plus env overrides pointing at the GCS paths.
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

    # Upload to GCS
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

    # Trigger Cloud Run Job
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
                        run_v2.EnvVar(name="FRAUD_RPM",    value="5"),
                        run_v2.EnvVar(name="FRAUD_VISION_CHAIN", value=os.environ.get("FRAUD_VISION_CHAIN", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash")),
                        run_v2.EnvVar(name="FRAUD_TEXT_CHAIN",   value=os.environ.get("FRAUD_TEXT_CHAIN", "gemini-3.7-flash,gemini-3.5-flash-lite,gemini-2.5-flash")),
                    ],
                    args=[],
                )
            ]
        ),
    )
    op = jobs_client.run_job(request=execution_request)
    execution_name = op.metadata.name  # projects/.../executions/EXEC-NAME

    return jsonify({
        "run_id":         run_id,
        "case_id":        case_id,
        "execution_name": execution_name,
        "files_uploaded": uploaded,
        "status":         "running",
    })


@app.route("/api/status")
@require_api_key
def status():
    """
    Query params:
      execution_name  — the full execution resource name returned by /upload
    """
    execution_name = request.args.get("execution_name", "")
    if not execution_name:
        return jsonify({"error": "execution_name is required"}), 400

    executions_client = run_v2.ExecutionsClient()
    execution = executions_client.get_execution(name=execution_name)

    # Map Cloud Run execution conditions to a simple state
    state = "running"
    if execution.succeeded_count and execution.succeeded_count > 0:
        state = "succeeded"
    elif execution.failed_count and execution.failed_count > 0:
        state = "failed"
    elif execution.cancelled_count and execution.cancelled_count > 0:
        state = "cancelled"

    return jsonify({
        "execution_name": execution_name,
        "state":          state,
        "succeeded":      execution.succeeded_count,
        "failed":         execution.failed_count,
        "start_time":     execution.start_time.isoformat() if execution.start_time else None,
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
