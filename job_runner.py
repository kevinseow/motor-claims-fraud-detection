"""
Cloud Run Job entrypoint for Motor Fraud Detection.

Orchestrates running single-case or batch fraud detection inside Cloud Run Jobs:
1. If GCS_BUCKET and GCS_RUN_ID are supplied (via UI trigger), downloads the case
   evidence from GCS (inputs/<run_id>/cases/) into /app/cases/.
2. Invokes run_fraud_detection.py with --case and other arguments.
3. Uploads generated results (FRAUD_DETECTION_RESULTS.md and raw JSON) back to
   GCS (outputs/<run_id>/) so the UI can retrieve them.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from google.cloud import storage

def main() -> int:
    bucket_name = os.environ.get("GCS_BUCKET")
    run_id = os.environ.get("GCS_RUN_ID")
    case_id = os.environ.get("GCS_CASE_ID") or os.environ.get("FRAUD_CASE_FILTER")
    
    print(f"=== Job Runner Started ===")
    print(f"GCS_BUCKET: {bucket_name}")
    print(f"GCS_RUN_ID: {run_id}")
    print(f"GCS_CASE_ID: {case_id}")
    print(f"CLI args: {sys.argv[1:]}")

    # 1. Pull input files from GCS if specified
    if bucket_name and run_id:
        print(f"Fetching input files from gs://{bucket_name}/inputs/{run_id}/ ...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        prefix = f"inputs/{run_id}/cases/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            rel_path = blob.name[len(prefix):]
            if not rel_path or rel_path.endswith("/"):
                continue
            dest_path = Path("/app/cases") / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest_path))
            print(f"  Downloaded: {dest_path.name} -> {dest_path}")

    # 2. Build the execution command
    cmd = [sys.executable, "run_fraud_detection.py"]
    if case_id:
        cmd.extend(["--case", case_id])
    
    # Forward any args passed to this container (e.g. --score)
    cmd.extend(sys.argv[1:])

    print(f"Running pipeline: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    # 3. Upload output files to GCS if specified
    if bucket_name and run_id:
        print(f"Uploading outputs to gs://{bucket_name}/outputs/{run_id}/ ...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Upload markdown report
        report_file = Path("FRAUD_DETECTION_RESULTS.md")
        if report_file.exists():
            out_blob = bucket.blob(f"outputs/{run_id}/FRAUD_DETECTION_RESULTS.md")
            out_blob.upload_from_filename(str(report_file))
            print(f"  Uploaded report: gs://{bucket_name}/outputs/{run_id}/FRAUD_DETECTION_RESULTS.md")

        # Upload raw JSONs
        raw_dir = Path("fraud_results_raw")
        if raw_dir.is_dir():
            for json_file in raw_dir.glob("*.json"):
                out_blob = bucket.blob(f"outputs/{run_id}/fraud_results_raw/{json_file.name}")
                out_blob.upload_from_filename(str(json_file))
                print(f"  Uploaded raw JSON: {json_file.name}")

    print(f"=== Job Runner Finished with exit code {result.returncode} ===")
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
