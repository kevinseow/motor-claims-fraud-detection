# Motor claim fraud detection — partner pack

A multimodal LLM reads a motor insurance claim pack (claim form PDF, repair
quotation PDF, police report where present, and damage photographs), applies
eight fraud detectors, and returns a verdict with a written rationale.

Everything needed is in this folder. One script, one command.

```
run_fraud_detection.py     the whole pipeline: prompt, reference tables, report writer
requirements.txt          pypdf, and pypdfium2 only for --doc-mode image
cases/                    13 anonymised claim packs (what the model sees)
reference/                price book + prior claims history (injected into the prompt)
ground_truth.json         real verdicts, for scoring only — see the warning below
```

## 🌐 Hosted Web UI (Cloud Deployment)

A web UI is deployed on Google Cloud Run for uploading motor claim packs and reviewing fraud assessment reports directly in your browser:

- **Live URL:** [https://fraud-ui-1082243885044.asia-southeast1.run.app](https://fraud-ui-1082243885044.asia-southeast1.run.app)
- **Access Key:** Configured securely via Secret Manager / environment variable (`UI_API_KEY`)
- **User Guide:** Detailed documentation in [fraud_ui/README.md](fraud_ui/README.md)

### Step-by-Step UI Guide:

| Step | Action | Description |
|---|---|---|
| **1. Authenticate** | Enter Access Key | Enter your authorized team access key. |
| **2. Case ID** *(optional)* | Set Case Reference | Type your claim reference (e.g., `CASE-155`) or leave blank to auto-generate a unique ID. |
| **3. Attach Files** | Drag & Drop Claim Pack | Drop documents (PDF) and damage photos (JPG/PNG). The UI automatically categorizes and validates them. |
| **4. Run Analysis** | Click "Analyse Pack" | The container runs Stage 1 (photo reading) and Stage 2 (fraud detector evaluation). |
| **5. Review Report** | View & Download | Inspect the fraud verdict, confidence, triggered rules (D1–D8), damage readings, and download as `.md`. |

> 💡 **Quick Test with Sample Cases:**  
> You can test the Web UI immediately using any sample folder in `cases/` (e.g. `cases/CASE-155/`):
> 1. Set Case Reference: `CASE-155`
> 2. Drag & drop `cases/CASE-155/form.pdf`, `cases/CASE-155/quotation.pdf`, and `cases/CASE-155/evidence/photo_01.jpg` into the upload zone.
> 3. Click **Analyse Claim Pack (3 files)** to watch the live progress and review the verdict.

---

## 💻 Quick start (Local CLI)

```bash
pip install -r requirements.txt

# against a local llama.cpp server
python run_fraud_detection.py --base-url http://127.0.0.1:8080/v1 --score

# against a hosted model
python run_fraud_detection.py \
    --base-url https://openrouter.ai/api/v1 \
    --api-key "$OPENROUTER_API_KEY" \
    --model some/strong-vision-model \
    --score
```

Outputs land next to the script:

| file | contents |
|---|---|
| `FRAUD_DETECTION_RESULTS.md` | the report: summary table, per-case rationale, detectors, scoring |
| `fraud_results_raw/CASE-nnn.json` | the model's raw JSON per case, plus token usage and timing |
| `fraud_detection_run.log` | every request and the model's full reply, including retries |

## Pointing it at a different model

Four settings, all available as flags or environment variables:

| flag | env | meaning |
|---|---|---|
| `--base-url` | `FRAUD_API_BASE` | base URL **including** the version path, e.g. `http://127.0.0.1:8080/v1` |
| `--api-key` | `FRAUD_API_KEY` | bearer token; a local server ignores it |
| `--model` | `FRAUD_MODEL` | model name as the endpoint knows it |
| `--flavor` | `FRAUD_API_FLAVOR` | `auto` (default), `llamacpp`, or `openai` |

`auto` picks `llamacpp` for a localhost URL and `openai` for anything else. The
only differences between the two dialects are authentication and two llama.cpp-only
request fields (`cache_prompt` and `chat_template_kwargs`), which a hosted endpoint
rejects with a 400. The chat format is identical either way: a system message plus
interleaved text and `image_url` parts carrying base64 data URIs.

So llama.cpp, vLLM, Ollama, LM Studio, OpenRouter and Gemini's OpenAI-compatible
shim all work without touching the code. If a hosted endpoint rejects the request,
try `--flavor llamacpp`; if a local one rejects `chat_template_kwargs`, try
`--flavor openai`.

Context window: the script asks llama.cpp for `n_ctx` and otherwise assumes 128k.
A local server with a small window will clamp `max_tokens` and log a warning.

## How it works

Two model calls per claim, because one call does not work well on a small model:

1. **Photo reading.** Each photograph is classified into the damage classes
   `crack`, `glass shatter`, `lamp broken`, `dent`, `scratch`, `tire flat`.
2. **Detector run.** Documents plus those readings, and no images, produce the
   verdict JSON.

The split matters. Photographs are tight close-up crops with no whole-vehicle shot,
so the only defensible question about them is *what kind of damage is this* — never
*which panel is this*. Isolating that into its own call stops the detector stage
from re-reading the images inconsistently and reasoning off the worse reading.

### Photo Deduplication & Caching
- **Content Hashing:** Computes SHA-256 hashes of all incoming vehicle photos.
- **Smart Cache Reuse:** If an identical photo was already analyzed in a previous claim or duplicate submission, the cached damage observations are reused instantly without making any Vision API call.
- **Storage:** Persisted locally in `photo_cache/` and in Google Cloud Storage (`gs://<BUCKET>/cache/photos/<sha256>.json`).

### Cloud Storage Lifecycle Management
- **30-Day Nearline Transition:** Claim packs in `gs://fraud-detection-511308-fraud-data` automatically transition from Standard to Nearline storage after 30 days (reducing storage costs by >70%).
- **90-Day Coldline Transition:** Moves to Coldline after 90 days for archival (>85% savings). Configured via `gcs_lifecycle.json`.

Other flags worth knowing:

```
--case CASE-155          one case (repeatable)
--doc-mode image        send rasterised PDF pages instead of extracted text
--photo-max-tokens N    stage-1 budget, raised automatically on a retry
--max-output-tokens N    stage-2 budget
--rescore               rebuild the report from saved JSON, no model calls
```

`--doc-mode text` (default) is roughly 6x faster and about half the tokens.
`--doc-mode image` treats the forms as pictures, which is a fairer test of a
vision model but needs `pypdfium2`.

## The eight detectors

| | technique | fires when |
|---|---|---|
| D1 | Ghost / invented damage | a claimed part has no photograph of a damage class it could exhibit |
| D2 | Severity inflation | declared severity is 2+ bands above observed **and** the estimate exceeds 4x market |
| D3 | Pre-existing or staged damage | narrative implies high energy, evidence is cosmetic |
| D4 | Invoice / parts inflation | quoted total exceeds 4x the market rate for the observed damage |
| D5 | Exclusion / policy not in force | loss date outside the policy period, or cover does not respond |
| D6 | Duplicate or repeat claim | prior claim on the same vehicle, same parts, predating this loss |
| D7 | Wear-and-tear claimed as collision | low-speed parking/kerb narrative on cosmetic-only damage |
| D8 | Loss-type mismatch | declared nature of loss admits none of the observed classes |

## Ground truth

**Operator-only. Delete `ground_truth.json` and this section from anything you
submit** — the whole dataset is anonymised so that no filename or document header
reveals the answer, and shipping the answers undoes that. The script never sends
it to the model; it is only read when you pass `--score`.

5 clean, 8 fraud. One case per technique, deliberately mixed across easy, medium
and hard.

| folder | original | verdict | technique | difficulty | case |
|---|---|---|---|---|---|
| `CASE-155` | `CLN-002` | clean | — | hard | Kerbside scrape, minor cosmetic only |
| `CASE-229` | `FRD-008` | fraud | D8 Loss-type mismatch (staged theft) | medium | Declared as theft, but the evidence shows impact damage |
| `CASE-278` | `CLN-001` | clean | — | easy | Rear-end collision at a signalised junction |
| `CASE-295` | `FRD-003` | fraud | D3 Pre-existing / staged damage | hard | Narrative describes a concrete-pillar impact; evidence is two surface scratches |
| `CASE-381` | `FRD-001` | fraud | D1 Ghost / invented damage | medium | Three of four claimed panels have no supporting photograph |
| `CASE-410` | `CLN-004` | clean | — | medium | Hail and wind damage to lamp and rear window |
| `CASE-502` | `CLN-005` | clean | — | medium | Tyre burst causing rollover — five panels, all photographed |
| `CASE-635` | `FRD-002` | fraud | D2 Severity inflation | easy | Two light scratches claimed as a write-off |
| `CASE-787` | `FRD-007` | fraud | D7 Wear-and-tear claimed as collision | hard | Five hairline scuffs presented as a road traffic accident |
| `CASE-822` | `FRD-006` | fraud | D6 Duplicate / repeat claim | medium | Same panels, same vehicle, resubmitted after an earlier settlement |
| `CASE-861` | `FRD-005` | fraud | D5 Exclusion / policy not in force | easy | Genuine collision on a policy that expired 11 days earlier |
| `CASE-872` | `CLN-003` | clean | — | hard | Car-park sideswipe with minor dent — wording resembles a wear claim |
| `CASE-947` | `FRD-004` | fraud | D4 Invoice / parts inflation | medium | Genuine moderate dent with a quotation 8x the market rate |

Declared facts per case, if you want to score by hand:

| folder | claimed parts | nature of loss | declared severity | declared estimate |
|---|---|---|---|---|
| `CASE-155` | Left Side Mirror | Collision | Minor cosmetic | 260.00 |
| `CASE-229` | Front Bumper, Front Left Headlamp | Theft | Moderate | 7,927.24 |
| `CASE-278` | Rear Bumper, Boot Lid | Collision | Moderate | 6,317.64 |
| `CASE-295` | Rear Left Wing, Rear Bumper | Collision | Moderate | 2,144.14 |
| `CASE-381` | Windscreen, Front Bumper, Rear Bumper, Front Left Headlamp | Collision | Moderate | 10,707.12 |
| `CASE-410` | Front Left Headlamp, Rear Window | Wind & Hail | Severe | 6,482.45 |
| `CASE-502` | Roof Panel, Front Left Wing, Rear Left Wing, Left Side Mirror, Front Left Tyre | Collision | Severe | 19,096.80 |
| `CASE-635` | Rear Left Door, Boot Lid | Collision | Vehicle not drivable | 14,800.00 |
| `CASE-787` | Front Bumper, Rear Bumper, Front Left Wing, Front Right Wing | Collision | Minor cosmetic | 4,147.22 |
| `CASE-822` | Front Bumper, Front Left Wing, Front Left Headlamp | Collision | Moderate | 8,360.08 |
| `CASE-861` | Front Bumper, Front Left Wing | Collision | Moderate | 8,832.27 |
| `CASE-872` | Front Right Wing | Collision | Minor cosmetic | 1,139.45 |
| `CASE-947` | Front Left Wing | Collision | Moderate | 3,955.07 |

Currency is SGD throughout.

### Three traps worth knowing before you score

- **The claim form's `DAMAGED PARTS` block is not a list of claims.** Every form
  prints the same catalogue of every panel the form offers, one per line, and the
  tick state does not survive PDF text extraction. The claimed parts are the
  **itemised quotation line items**. Reading the form block as the claim makes you
  report ghost damage on every clean case.
- **`EXTENT OF DAMAGE` has the same problem** — `Minor cosmetic`, `Moderate`,
  `Severe` and `Vehicle not drivable` are all printed. Cross-check the declared
  severity against the estimated loss.
- **Photographs are crops, not panels.** Support is decided by whether the visible
  damage *class* is one the claimed part could exhibit. Guessing the panel from a
  crop produces both false ghosts on real claims and misses on invented ones.

## Measured results, and what is still wrong

These are from `gemma-4-26B-A4B-it-UD-IQ3_XXS` (a 3.06 bpw quantisation) run
fully offloaded on one RTX 4060 Ti (16 GB) via llama.cpp, temperature 0, all 13
cases, ~15 minutes:

| metric | result |
|---|---|
| fraud-call accuracy (fraud or not) | **10/13** |
| exact verdict match | 10/13 |
| ground-truth technique among those fired | 6/8 |
| false positives (clean called fraud) | 3 — `CASE-278`, `CASE-410`, `CASE-502` |
| false negatives (fraud called clean) | 0 |

Treat this as a baseline for a small quantised model, not a ceiling. The known
weaknesses, in the order I would attack them:

1. **False positives on clean cases.** All three fire a detector that should stay
   quiet: D4 on a genuinely market-priced repair, D8 on a clerical
   declared-vs-narrative mismatch, D1 on a five-panel rollover where the photo
   reading was degraded. Detector thresholds are tuned too loose.
2. **Two techniques never fire.** D7 (wear-and-tear) and D5 (coverage gate). On
   `CASE-861` the model correctly computed `in_force: false` in its coverage check
   and then failed to mark D5 as fired — the reasoning was right and the encoding
   was not. Both are pure text tasks that a stronger model should get right.
3. **Stage-1 photo reading is the bottleneck.** Reasoning on a crop sometimes ran
   to 8,000+ tokens without answering. There is an automatic retry ladder
   (bigger budget, then reasoning off) and it recovers every case, but the
   reasoning-off fallback produces visibly worse class readings, which then
   poisons D1. A stronger model should need none of this.

A model with reliable vision should mostly fix 1 and 3, and fix 2 outright.
Re-run with `--score` and compare against the table above.