# Motor Claims Fraud Detector — Web UI Guide & User Manual

A cloud-hosted, multimodal AI web application that analyzes motor insurance claim packs (claim form PDFs, repair estimates/quotations, police reports, and damage photographs) to detect fraud indicators.

---

## 🌐 Live Web Application

| Parameter | Details |
|---|---|
| **App URL** | [https://fraud-ui-1082243885044.asia-southeast1.run.app](https://fraud-ui-1082243885044.asia-southeast1.run.app) |
| **Access Key** | Configured via Secret Manager (`UI_API_KEY`) |
| **Cloud Hosting** | Google Cloud Run (Serverless, scales to 0 when idle) |
| **Region** | `asia-southeast1` (Singapore) |

---

## 🚀 Quick Step-by-Step Walkthrough

```
[1. Open UI] ──► [2. Enter Access Key & Case ID] ──► [3. Drag & Drop Files] ──► [4. Click "Analyse Pack"] ──► [5. View / Download Report]
```

### Step 1: Open the App & Authenticate
1. Navigate to **[https://fraud-ui-1082243885044.asia-southeast1.run.app](https://fraud-ui-1082243885044.asia-southeast1.run.app)**.
2. In the **Access Key** field, enter your assigned authorization key.

### Step 2: Set the Case Reference
* Enter your internal claim reference (e.g. `CASE-155` or `CLM-2026-9810`).
* *(Optional)* If you leave this field empty, a unique reference ID (e.g. `CASE-A1B2C3`) is generated automatically.

### Step 3: Attach Claim Documents & Evidence Photos
Drag and drop (or click to browse) all files belonging to the claim:
* **Documents (`.pdf`):**
  * Completed Claim Form (loss description, dates, policyholder declaration)
  * Repairer Quotation / Estimate (itemized parts and labour rates)
  * Police Report or Survey Report (if available)
* **Photographs (`.jpg`, `.jpeg`, `.png`):**
  * Vehicle damage photographs
  * Accident scene photos
  * Odometer and license plate photos

> **Automatic File Organization:**  
> You do not need to separate folders. The backend automatically places `.pdf` documents in the case root and images into the `evidence/` subfolder required by the vision pipeline.

### Step 4: Click "Analyse Pack"
* The files are uploaded to Google Cloud Storage (`gs://fraud-detection-511308-fraud-data/inputs/<run_id>/`).
* The system triggers the background Cloud Run detection job (`fraud-detection-job`).
* A live progress bar and status tracker will monitor the analysis until completion.

### Step 5: Review & Download the Report
Once complete:
* The full report renders directly in the browser with:
  * **Summary Table:** Verdict, Fraud Probability (0–100%), Confidence, Primary Detector, Key Finding, and Recommended Action.
  * **Photographic Evidence:** Damage observations (`dent`, `scratch`, `crack`, `glass shatter`, `lamp broken`, `tire flat`) identified per photograph.
  * **Detectors Triggered (D1–D8):** Which specific fraud rules were violated.
  * **Red Flags & Counter-Evidence:** Inconsistencies vs. genuine indicators.
  * **Recommended Next Steps:** Practical actions for adjusters/investigators.
  * **Analyst Rationale:** Detailed narrative explaining the decision.
* Click **⬇ Download .md** to save the complete report locally.

---

## 🧪 Testing with Included Sample Cases

The repository includes 13 anonymized sample claim packs in the `cases/` directory. You can test the web app immediately using any of them:

### Sample: `CASE-155`
1. In the Web UI, enter Case Reference: `CASE-155`
2. Drag and drop these files from `cases/CASE-155/`:
   * `cases/CASE-155/form.pdf`
   * `cases/CASE-155/quotation.pdf`
   * `cases/CASE-155/evidence/photo_01.jpg`
3. Click **Analyse Pack**.

---

## 🧠 AI Pipeline & Automatic Multi-Model Fallback

The pipeline uses a two-stage approach with **automatic multi-model fallback chains** to maximize reliability on the free tier:

```
Claim Pack
    │
    ├──► Stage 1: Photo Reading (Vision Model Chain)
    │       Reads every photograph for physical damage classes.
    │       Chain: gemini-3.8-flash ──(fallback)──► gemini-3.7-flash ──(fallback)──► gemini-3.6-flash
    │
    └──► Stage 2: Fraud Detectors (Text Model Chain)
            Synthesizes documents, photo findings & reference rules D1–D8.
            Chain: gemini-3.7-flash ──(fallback)──► gemini-3.5-flash-lite ──(fallback)──► gemini-2.5-flash
```

* **5 RPM Smooth Pacing:** Calls are paced with a minimum 12-second interval to eliminate burst 429 errors.
* **Quota Failover:** If a primary model reaches its daily limit, the pipeline automatically fails over to the next backup model with zero interruption.

---

## ⚡ Cloud Tasks Decoupled Queue & Rate Limiting

To scale smoothly under high concurrency without dropping requests or triggering 429 quota exhaustion:

```
[Claim Upload] ──► [GCS Ingestion] ──► [Cloud Tasks Queue] ──► [Worker Endpoint] ──► [Cloud Run Job]
 (User Request)      (Inputs Stored)    (Rate-Limited Buffer)   (Dispatched 10/s)      (Multimodal Run)
```

* **Queue Name:** `projects/fraud-detection-511308/locations/asia-southeast1/queues/fraud-claims-queue`
* **Rate Limits:**
  * `max_dispatches_per_second`: `10`
  * `max_concurrent_dispatches`: `20`
* **Retry Policy with Exponential Backoff:**
  * `max_attempts`: `5`
  * `min_backoff`: `5s`
  * `max_backoff`: `60s`
  * `max_doublings`: `3`

---

## 🛡️ Detectors Evaluated (D1–D8)

| Code | Detector | What It Evaluates |
|---|---|---|
| **D1** | Damage Severity & Nature | Physical damage matches declared incident mechanics |
| **D2** | Labour & Consumables Overhead | Repairer labour and overhead lines match market standards |
| **D3** | Incompatible Loss Classes | Declared nature of loss supports claimed damage types |
| **D4** | Part Price Inflations | Itemized part prices compared against market price book |
| **D5** | Damage Severity Ratio | Multiplier of declared severity vs. observed damage share |
| **D6** | Prior Claims Overlap | Checks claimed parts against vehicle's prior loss history |
| **D7** | Timing & Policy Inconsistencies | Loss timing relative to policy inception / renewal |
| **D8** | Vehicle Age & Market Value Disproportion | Total repair estimate relative to vehicle market value |

---

## 🛠️ How to Redeploy Updates

```powershell
# Rebuild the UI container
gcloud builds submit ./fraud_ui `
    --tag "asia-southeast1-docker.pkg.dev/fraud-detection-511308/fraud-detection/ui:latest" `
    --project fraud-detection-511308 --quiet

# Deploy updated Cloud Run service
gcloud run deploy fraud-ui `
    --image="asia-southeast1-docker.pkg.dev/fraud-detection-511308/fraud-detection/ui:latest" `
    --region="asia-southeast1" `
    --project="fraud-detection-511308" --quiet
```
