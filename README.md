# Motor Claims Fraud Detector

An enterprise multimodal AI system for automated motor insurance claim fraud detection. It analyzes claim packs (claim form PDFs, repair estimates/quotations, police/survey reports, and vehicle damage photographs), evaluates eight specialized fraud detectors (D1–D8), and generates an adjuster-ready investigation report with transparent reasoning.

---

## 🌐 Live Web Application

The application is deployed on Google Cloud Platform and accessible via a modern web interface:

| Parameter | Details |
|---|---|
| **App URL** | [https://fraud-ui-1082243885044.asia-southeast1.run.app](https://fraud-ui-1082243885044.asia-southeast1.run.app) |
| **Authentication** | Access Key configured securely via Google Secret Manager (`UI_API_KEY`) |
| **Hosting** | Google Cloud Run (Fully serverless, autoscaling, scales to zero when idle) |
| **Region** | `asia-southeast1` (Singapore) |

---

## 🚀 Step-by-Step Web UI Guide

```
┌──────────────┐     ┌──────────────────────┐     ┌─────────────────────┐     ┌─────────────────┐     ┌────────────────────┐
│ 1. Open App  │ ──► │ 2. Enter Access Key  │ ──► │ 3. Drag & Drop Pack │ ──► │ 4. Click Run   │ ──► │ 5. Review & Export │
│ & Dashboard  │     │    & Case Reference  │     │    (PDFs & Photos)  │     │    "Analyse"    │     │    (Report / .md)  │
└──────────────┘     └──────────────────────┘     └─────────────────────┘     └─────────────────┘     └────────────────────┘
```

### 1. Authenticate
Open **[https://fraud-ui-1082243885044.asia-southeast1.run.app](https://fraud-ui-1082243885044.asia-southeast1.run.app)** and enter your assigned **Access Key** into the authentication field.

### 2. Set Case Reference *(Optional)*
Enter your internal claim identifier (e.g. `CASE-155` or `CLM-2026-9810`). If left blank, a unique reference ID (e.g. `CASE-A1B2C3`) is generated automatically.

### 3. Attach Claim Documents & Evidence Photos
Drag and drop (or click to browse) all files belonging to the claim:
* **Documents (`.pdf`):**
  * Completed Claim Form (loss description, dates, driver/insured statement)
  * Repairer Quotation / Estimate (itemized parts and labour rates)
  * Police Report or Independent Survey Report (if available)
* **Photographs (`.jpg`, `.jpeg`, `.png`):**
  * Vehicle damage photographs
  * Accident scene photos
  * Odometer and license plate photos

> **Automatic File Routing:**  
> The backend automatically categorizes `.pdf` documents to the claim root and routes images to the `evidence/` subfolder required by the vision pipeline.

### 4. Click "Analyse Pack"
The UI uploads the files to Google Cloud Storage (`gs://fraud-detection-511308-fraud-data/inputs/<run_id>/`) and dispatches the job to the backend runner. A live progress tracker displays real-time execution status.

### 5. Review & Export Findings
Once analysis completes, the browser presents:
* **Verdict Summary:** Overall Fraud Verdict (`clean` / `fraud`), Fraud Probability (0–100%), Confidence, and Primary Detector.
* **Photographic Evidence:** Damage observations (`dent`, `scratch`, `crack`, `glass shatter`, `lamp broken`, `tire flat`) identified per photograph.
* **Detectors Triggered (D1–D8):** Specific fraud rules flagged with underlying evidence.
* **Red Flags & Counter-Evidence:** Inconsistencies vs. genuine indicators.
* **Recommended Next Steps:** Actionable guidance for claims adjusters and Special Investigation Units (SIU).
* **Export:** Click **⬇ Download .md** to save the markdown report.

---

### 🧪 Testing with Included Sample Cases

The repository includes 13 anonymized sample claim packs in the `cases/` directory:

1. In the Web UI, enter Case Reference: `CASE-155`
2. Drag and drop these files from `cases/CASE-155/`:
   * `cases/CASE-155/form.pdf`
   * `cases/CASE-155/quotation.pdf`
   * `cases/CASE-155/evidence/photo_01.jpg`
3. Click **Analyse Pack** to test live execution.

---

## ⚡ Cloud Architecture & Scaled Production Pattern

The system decouples user ingestion from heavy AI inference to handle high concurrency smoothly without dropping requests or encountering API rate-limit errors:

```
[User Browser]
      │
      ▼
[Cloud Run: fraud-ui] ──── Uploads Files ────► [Cloud Storage: gs://...-fraud-data]
      │                                                │
      ▼ Enqueues Task                                  │ Reads Claim Pack
[Cloud Tasks Queue: fraud-claims-queue]                │
      │                                                ▼
      └──────── Dispatches (Rate Limited) ────► [Cloud Run Job: fraud-detection-job]
                                                       │
                                                       ├──► Stage 1: Photo Reading (Vision Chain)
                                                       ├──► Stage 2: Fraud Evaluation (Text Chain)
                                                       └──► Writes JSON & Markdown Reports
```

### 1. Cloud Tasks Decoupled Queue & Rate Limiting
* **Queue Name:** `projects/fraud-detection-511308/locations/asia-southeast1/queues/fraud-claims-queue`
* **Concurrency & Pacing:**
  * `max_dispatches_per_second`: `10`
  * `max_concurrent_dispatches`: `20`
* **Automatic Retries with Exponential Backoff:**
  * `max_attempts`: `5`
  * `min_backoff`: `5s`
  * `max_backoff`: `60s`
  * Automatically recovers from transient network hiccups or model timeouts without failing claims.

### 2. Photo Deduplication & SHA-256 Caching
* **Cryptographic Content Hashing:** Computes SHA-256 hashes of all incoming vehicle photographs.
* **Zero-Cost Cache Hits:** If an identical photo was already analyzed in a previous submission or another claim, cached damage observations (`classes`, `severity`, `share`, `description`) are reused immediately.
* **Cross-Claim Duplicate Detection:** Identifies when the exact same photograph is submitted across different case numbers—directly flagging duplicate claim fraud.
* **Hybrid Storage:** Cached locally (`photo_cache/`) and in Google Cloud Storage (`gs://<BUCKET>/cache/photos/<sha256>.json`).

### 3. Cloud Storage Lifecycle Management
Managed via [gcs_lifecycle.json](gcs_lifecycle.json) on `gs://fraud-detection-511308-fraud-data`:
* **30-Day Nearline Transition:** Claim inputs and analysis outputs automatically transition from Standard to Nearline storage after 30 days (**>70% cost reduction**).
* **90-Day Coldline Transition:** Transitions to Coldline storage after 90 days for long-term audit and compliance archiving (**>85% cost reduction**).

---

## 🧠 Multimodal AI Pipeline & Multi-Model Fallback

AI inference is isolated into two stages to prevent small vision models from hallucinating vehicle panel locations:

```
Claim Pack
    │
    ├──► Stage 1: Photo Reading (Vision Model Chain)
    │       Reads every photo for physical damage classes.
    │       Chain: gemini-3.8-flash ──(fallback)──► gemini-3.7-flash ──(fallback)──► gemini-3.6-flash
    │
    └──► Stage 2: Fraud Detectors (Text Model Chain)
            Synthesizes documents, photo findings & reference rules D1–D8.
            Chain: gemini-3.7-flash ──(fallback)──► gemini-3.5-flash-lite ──(fallback)──► gemini-2.5-flash
```

* **Smooth Rate Pacing:** Paced with minimum 12-second intervals (~5 RPM) to stay within free-tier quotas.
* **Automatic Quota Failover:** If a primary model encounters rate limits (HTTP 429) or quota exhaustion, it immediately falls back to the secondary and tertiary backup models with zero manual intervention.

---

## 🛡️ The Eight Fraud Detectors (D1–D8)

| Code | Detector | What It Evaluates | Trigger Condition |
|---|---|---|---|
| **D1** | Ghost / Invented Damage | Claimed parts vs. photographic evidence | A claimed part has no photograph of a damage class it could exhibit |
| **D2** | Severity Inflation | Declared severity vs. observed damage | Declared severity is 2+ bands above observed **and** estimate exceeds 4x market |
| **D3** | Pre-Existing / Staged Damage | Impact mechanics vs. physical marks | Loss narrative implies high energy, but evidence shows only superficial/cosmetic marks |
| **D4** | Part Price & Invoice Inflation | Quotation prices vs. market price book | Quoted repair total exceeds 4x the market rate for the observed damage |
| **D5** | Policy Inception / Exclusion Gate | Coverage validity and loss timeline | Incident date falls outside active policy coverage dates |
| **D6** | Duplicate / Repeat Claim | Prior claims history overlap | Identical parts/vehicle claimed and settled in a prior claim |
| **D7** | Wear-and-Tear as Collision | Age-related deterioration vs. impact | Low-speed collision narrative presented for cumulative wear/scuffs |
| **D8** | Loss-Type Mismatch | Declared incident type vs. damage nature | Declared nature of loss admits none of the observed damage classes |

---

## 🔍 Critical Analysis Traps to Avoid

1. **Claim Form Checkboxes vs. Quotation Items:**
   * The PDF claim form prints a master list of all possible vehicle panels; checkboxes often flatten during PDF text extraction.
   * **Rule:** The claimed parts are strictly determined by the **itemized quotation line items**, not the form's template options.
2. **Close-Up Crops vs. Whole-Vehicle Panels:**
   * Claim photos are tight close-up crops without full-vehicle context.
   * **Rule:** Photographs only answer *what damage class is present* (`dent`, `scratch`, `crack`), not *which panel it belongs to*.
3. **Declared Extent vs. Estimated Loss:**
   * Forms print all severity levels (`Minor cosmetic`, `Moderate`, `Severe`). Cross-check narrative severity against the repairer estimate and market baseline.

---

## 💻 Local CLI Execution

You can run the entire pipeline locally against any OpenAI-compatible API or local model runner:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with Gemini (OpenAI-compatible endpoint)
python run_fraud_detection.py \
    --base-url "https://generativelanguage.googleapis.com/v1beta/openai" \
    --api-key "$GEMINI_API_KEY" \
    --score

# 3. Run against local llama.cpp
python run_fraud_detection.py \
    --base-url "http://127.0.0.1:8080/v1" \
    --score

# 4. Run a single case
python run_fraud_detection.py --case CASE-155 --score
```

### Useful CLI Flags:
```
--case <CASE_ID>         Process a specific case (repeatable)
--doc-mode text|image    Use extracted text (fast, default) or rasterized PDF pages (image)
--photo-max-tokens N     Stage 1 token budget (auto-escalates on retry)
--max-output-tokens N    Stage 2 token budget
--rescore                Rebuild report from saved JSON without re-running models
```

---

## 🛠️ Cloud Deployment & Redeployment

### Rebuild and Deploy the Web UI:
```powershell
# Rebuild the UI container
gcloud builds submit ./fraud_ui `
    --tag "asia-southeast1-docker.pkg.dev/fraud-detection-511308/fraud-detection/ui:latest" `
    --project fraud-detection-511308 --quiet

# Deploy to Cloud Run
gcloud run deploy fraud-ui `
    --image="asia-southeast1-docker.pkg.dev/fraud-detection-511308/fraud-detection/ui:latest" `
    --region="asia-southeast1" `
    --project="fraud-detection-511308" --quiet
```

### Rebuild the Analysis Runner:
```powershell
gcloud builds submit . `
    --config cloudbuild.yaml `
    --project fraud-detection-511308 --quiet
```