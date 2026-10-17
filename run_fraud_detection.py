"""Motor claim fraud detection against a local or hosted vision LLM.

One file. It contains the detection prompt (D1-D8), the reference tables, the
claim-pack reader and the report writer. Copy the whole folder and run it.

    pip install -r requirements.txt
    python run_fraud_detection.py --score

Talking to a different model
----------------------------
Everything provider-specific is behind four settings. Either pass them as flags
or set the environment variables:

    --base-url   (FRAUD_API_BASE)  e.g. http://127.0.0.1:8080/v1
    --model      (FRAUD_MODEL)     e.g. gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf
    --api-key    (FRAUD_API_KEY)   ignored by a local server
    --flavor     (FRAUD_API_FLAVOR)  auto | llamacpp | openai

`--flavor auto` (the default) picks `llamacpp` for a localhost base URL and
`openai` for anything else. The only real differences are authentication and two
llama.cpp-only request fields (`chat_template_kwargs`, `cache_prompt`), so any
OpenAI-compatible endpoint -- llama.cpp, vLLM, Ollama, LM Studio, OpenRouter,
Gemini's OpenAI shim -- works without code changes. The chat format is the same
either way: a system message plus interleaved text and `image_url` parts holding
base64 data URIs.

    # local llama.cpp
    python run_fraud_detection.py --base-url http://127.0.0.1:8080/v1

    # a hosted model
    python run_fraud_detection.py \
        --base-url https://openrouter.ai/api/v1 \
        --api-key "$OPENROUTER_API_KEY" \
        --model some/strong-vision-model

The pipeline is two calls per claim, because a single call does not work well:

  1. photo reading  -- every photograph is classified into the damage classes
     `crack`, `glass shatter`, `lamp broken`, `dent`, `scratch`, `tire flat`.
     Reasoning on. This call tolerates a generous token budget.
  2. detector run   -- documents plus those readings, and no images. Returns the
     verdict JSON. Reasoning off, because reasoning on this task tends not to
     terminate.

Stronger models usually want stage 1 only, since they read photographs reliably
on the first attempt; if a hosted model rejects `--flavor openai` payloads or
the reasoning toggle is unsupported, that is the switch to flip.

Ground truth
------------
`ground_truth.json` maps each anonymised folder to its real verdict. It is for
scoring only, is never sent to the model, and must be deleted from any bundle
you submit -- see the warning in README.md.
"""

from __future__ import annotations


import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference"

DAMAGE_CLASSES = ("crack", "glass shatter", "lamp broken", "dent", "scratch", "tire flat")

LOSS_TYPE_COMPATIBILITY = {
    "Collision": ["crack", "glass shatter", "lamp broken", "dent", "scratch", "tire flat"],
    "Theft": [],
    "Fire": [],
    "Flood": [],
    "Wind & Hail": ["crack", "glass shatter", "lamp broken", "dent", "scratch"],
    "Glass Damage": ["crack", "glass shatter"],
    "Third Party Property Damage": ["dent", "scratch", "crack", "glass shatter", "lamp broken"],
    "Found Damaged": ["crack", "glass shatter", "lamp broken", "dent", "scratch", "tire flat"],
}

ROLE = """\
You are a senior motor claims assessor in Singapore (SGD, right-hand traffic, LTA
registration format). You review motor claims for possible fraud and write a
referral note for a human claims manager.

You are an investigator, not a classifier. A finding is only defensible if you
can name the document line or photograph that supports it. "The numbers look odd"
is not a finding; "the narrative describes a reversing manoeuvre that could not
have produced the damage photographed" is. Most claims are genuine. Never
manufacture a finding because the categories below exist.
"""

INPUTS = """\
# The pack in front of you

1. **Claim form** -- the claimant's own account: accident narrative, ticked nature
   of loss, ticked damaged parts, declared extent of damage, declared estimate,
   policy period, driver identity.
2. **Repair quotation** -- the workshop's itemised invoice. A THIRD-PARTY document.
   A workshop can inflate a bill without the policyholder knowing, so if the
   inflation is in the quotation rather than the form, say so: that changes the
   action from "investigate the policyholder" to "refer the repairer".
3. **Police report** -- present only sometimes. Absence is NOT suspicious; genuine
   low-speed car-park scrapes routinely have no report.
4. **Evidence photographs** -- the claimant's damage photos.
5. **Reference data** -- price book, severity calibration, prior claims history,
   supplied below.
"""

EVIDENCE_RULES = """\
# Rules of evidence

- The claim form's `DAMAGED PARTS` block is a **printed catalogue of every part the
  form offers**, one per line, with no mark against the ones not claimed. It is NOT
  the list of parts this claimant is claiming, and reading it as such makes you
  report ghost damage on every clean claim. The claimed parts are the **itemised
  quotation line items** -- one line per panel, e.g. `Front Bumper - dent / scratch
  repair`. Use the quotation for the part list. Where a quotation line describes the
  work (`replacement panel`, `mirror glass replacement`, `panel pull and hammer
  work`), that is the repairer's characterisation, not proof; the photographs still
  decide support.
- The claim form's `EXTENT OF DAMAGE` block has the same problem: `Minor cosmetic`,
  `Moderate`, `Severe` and `Vehicle not drivable` are all printed. The declared
  severity is the one the claimant marked, cross-check against the estimated loss.
- Check coverage before anything else and record the result in `coverage_check`
  first: compare the loss date against the policy period printed on the form, the
  declared nature of loss against the cover type, and the driver against the insured.
  If `in_force` is false then D5 fires and the verdict is `not_covered` -- those three
  facts are all in the documents and no photograph is needed to reach them. Do not
  conclude in `reason_not_fired` that the date falls outside the policy and then
  leave D5 unfired.
- Work only from what is in front of you. If a document does not say it, it does
  not say it.
- The photographs are tight close-ups of damaged areas with no whole-vehicle shot,
  so you **cannot** tell which panel any photo shows. Test *class consistency* --
  "does this photo show a kind of damage this part could exhibit?" -- and never
  assert panel identity. "photo_02.jpg shows a dent, a class the front bumper can
  exhibit" is an observation; "photo_02.jpg shows the front bumper" is a
  fabrication.
- Your own guess at *which* panel a photo shows is never evidence. Write it, but
  then ignore it. Support is decided by the class table and nothing else: if a
  claimed part accepts `crack` and any photo shows `crack`, the part is SUPPORTED
  even if you believe the photo is of some other panel. Do not let an uncertain
  panel guess turn a class match into a contradiction -- that single substitution
  produces both false ghosts on real claims and misses on invented ones.
- Support is a lookup, not a judgement. For each claimed part: look up its accepted
  classes in the part table, then look up the classes you observed in the photos.
  If the two sets intersect at all, the part is SUPPORTED. If they do not
  intersect, the part is UNSUPPORTED **only if the pack was adequate** -- otherwise
  UNASSESSABLE. Report the lookup in `part_support` with the class that decided it.
- Damage classes are exactly: {classes}.
- **Classify every photograph before reasoning about anything.** Per photo state
  the classes visible, the rough share of the frame damaged, and one line of
  description. Do not merge or skip photographs.
- If a photo is too small, blurred or ambiguous, mark it `unclassifiable` and say
  why. Never invent a class to fill a gap; an unclassifiable photo is neither
  support nor contradiction.
- A part is UNSUPPORTED only when the pack was adequate to show it and did not.
  Two photos attached to a claim for five parts is a documentation problem, not
  evidence of fraud.
- Folder names, case references and claim references are administrative only and
  were randomised. They carry no information about the outcome. Do not reason
  from them and never treat an identifier as corroboration.
"""

DETECTORS = """\
# The eight detectors

Report on all eight. For those that stay quiet give the specific reason -- "D4
quiet: quoted total is 1.0x market at the observed band" -- because an assessor
who can say why a detector stayed quiet is worth more than one who only reports
hits.

**D1 Ghost / invented damage.** Mechanical test, no interpretation. For each part
ticked on the form, read its accepted damage classes from the part table and check
whether any photo shows at least one of them. If the sets intersect, the part is
supported and D1 stays quiet for it -- full stop, whatever panel you think the photo
shows. Only a part with an EMPTY intersection against an adequate pack is a ghost.
Zero supporting photos across a multi-part claim is the finding: name the parts and
give the count. Strongest form is a part whose ONLY accepted class appears nowhere in
the pack (a headlamp with no `lamp broken` anywhere; a tyre with no `tire flat`).
Strengthens when the declared severity is high or the car is declared undriveable
while the evidence is minor, or when the narrative describes an impact that would
physically have deformed exactly the unphotographed panels.

**D2 Severity inflation.** Map observed damage extent to a band with the
calibration table. Fire only when BOTH hold: the declared severity sits 2+ bands
above your observed band, AND the declared estimate exceeds 4.0x the market
expectation priced at the OBSERVED band. Direction-sensitive, not
magnitude-sensitive: minor damage with a minor estimate is normal, not a finding.
Low observed damage with a high estimate is also legitimate when structural
damage hides behind cosmetic damage and close-up photos cannot see it. Referral
trigger, never a decline.

**D3 Pre-existing or staged damage.** Fire when the narrative asserts a
high-energy or specific impact ("at speed", "hit", "spun", "concrete pillar",
"drifted off the road") but the evidence contains only cosmetic surface damage --
scratch or minor dent -- with no deformation, crush-zone evidence, glass failure
or lamp breakage. Strengthens when declared severity exceeds what the photos show,
or a panel-replacement line is quoted against cosmetic-only evidence. Honest
limit: you cannot see paint-meter readings, corrosion or panel manufacturing dates.
Report the narrative-versus-evidence contradiction only; never claim to have
detected paint mismatch.

**D4 Invoice or parts inflation.** Magnitude test: price the market expectation
(recipe below), then `ratio = quoted total / market expectation`, fire above 4.0x.
Composition test, fires independently of magnitude: a watchlist category the
observed damage does not justify (calibration, sensor, premium finish, paint
blending, accessory -- none of which a scratch requires); full panel REPLACEMENT
quoted where the price book lists a materially cheaper repair line for scratch-only
evidence; parts billed that are not ticked on the form at all. Attribute to the
repairer unless the claim form carries the false figure, and say which.

**D5 Exclusion / policy not in force.** Pure rules layer, no images needed. The
date of accident must fall inside the policy period printed on the form; the
cover type must respond to the declared nature of loss; the driver must be the
insured or a permitted driver. This is a GATE, not a score, reported in its own
field rather than folded into the fraud probability: if the policy was not in
force the claim cannot proceed however genuine the damage is. It is overwhelmingly
clerical rather than dishonest, so set `verdict` to `not_covered`, keep the fraud
  probability low, and recommend a coverage referral. Do not call the policyholder a
  fraudster on this ground. `in_force: false` in `coverage_check` means D5 fired --
  there is no reading of the evidence in which that check fails and D5 stays quiet.

**D6 Duplicate or repeat claim.** Look the registration and chassis up in the
prior claims history below. Fire when a prior claim on the same vehicle covers
substantially the same parts, PREDATES this loss date, AND those parts are
supported by evidence in this claim's photographs too. That last condition matters:
if this claim's parts have no photographic support, a prior-claim match is far
more likely an earlier genuine repair than a duplicate, and a ghost-part finding
(D1) is the better finding. Check ordering -- a prior claim closed before this
loss date is mostly administrative; one still open, or settled for the same parts
at a different workshop, is stronger.

**D7 Wear-and-tear claimed as collision.** Fire only when all three hold: the
narrative describes a parking manoeuvre, reversing, a kerb or wall at low speed,
"noticed on return" or "found damaged"; the observed damage is cosmetic-only and
minor; and the loss is nonetheless declared as a collision against a third party.
The discriminator is the ABSENCE of corroboration, never the phrase. Corroboration
is any one of: a named third party, a named witness, a police report. If one
exists and the damage is real, do not fire -- genuine car-park sideswipes use
exactly this vocabulary. Phrase matching alone must never produce this finding.

**D8 Loss-type mismatch (staged theft).** Use the compatibility matrix below. Fire
when the declared loss type admits NONE of the classes you actually observed.
Honest limit: a genuine theft can leave incidental impact damage from dragging
the vehicle or broken glass, so this is a mismatch requiring investigation, never
a conclusion, and on its own it caps the fraud probability at 70.
"""

RECIPE = """\
# Pricing recipe (D2 and D4)

Per part on the claim form: take the rate line matching the damage you actually
observed -- replacement for deformation or glass/lamp failure, REPAIR for a
scratch or light dent -- add the refinish rate for painted panels, multiply by
the severity multiplier for your OBSERVED band. Add the consumables overhead once
for the job. That is the market expectation; compare it to the quoted total and
separately to the declared estimate. If the quotation total and the form estimate
disagree with each other, that is a finding on its own -- report it.
"""

DISCIPLINE = """\
# Calibration and false-positive discipline

Read this before scoring; it decides whether you are useful.

- **Assume most claims are genuine.** Referring a pack requires a specific,
  nameable contradiction. If your honest conclusion is "the pack is thin but
  nothing is wrong", say that and score it low. A defensible low score beats a
  confident guess.
- **Narrative keywords are weak by construction.** Genuine claimants describe
  parking manoeuvres; fraudsters do too. Corroboration is the discriminator.
- **Name one primary finding.** A claim may legitimately trip two detectors when
  two problems are genuinely present -- report both, designate one as primary,
  and say why it dominates.
- **Two detectors on independent evidence agreeing beats one firing loudly.**
  Add 10-15 points for agreement; one detector alone caps the probability at 85.
- **Never auto-decline.** Every case goes to a human. Say what you would do next.
- Fraud probability bands: 90-100 a specific verifiable contradiction in the pack
  itself (a date gate, arithmetic, a named part with no possible supporting
  evidence); 70-89 one strong inconsistency on a single channel; 45-69 a
  suggestive pattern needing a source the pack does not contain; 20-44 weak,
  noted but not relied on; 0-19 nothing supportable, an ordinary claim.
"""

OUTPUT = """\
# Required output

Reply with ONE JSON object and nothing else -- no prose before or after it, no
markdown code fence. Fields:

- `case_reference` (string) -- the identifier you were given.
- `verdict` (string) -- `clean` | `fraud` | `needs_investigation` | `not_covered`,
  chosen by this rule and no other:
  - `not_covered` if and only if the coverage gate failed (D5): the policy was not in
    force at the loss date, or the cover type does not respond to the declared loss.
  - `fraud` if any detector fired on evidence you can quote from the pack. A confirmed
    ghost part, a confirmed duplicate, a confirmed exclusion or a quote far above
    market IS the finding -- it does not become an investigation just because the
    pack is small.
  - `needs_investigation` ONLY when you can name what is wrong but cannot tie it to
    evidence in the pack: an unexplained discrepancy, a detection resting on a photo
    you marked `unclassifiable`, a pattern that needs records you do not have.
  - `clean` when nothing fires.
  `needs_investigation` is not a softer `fraud`. Never recommend escalating to SIU
  while returning `needs_investigation` on a finding you called high confidence --
  that contradiction is the most common way this report gets rejected. If the honest
  answer is "an assessor should look again", the probability sits at 40-65 and the
  verdict is `fraud` only when a detector actually fired on quoted evidence.
- `fraud_probability` (int 0-100) -- calibrated probability the claim is
  fraudulent. Must agree with `verdict`: `clean` low, `fraud` high, `not_covered`
  low (the damage may be genuine; the policy simply did not respond).
- `confidence` (string) -- `low` | `medium` | `high`, in your confidence in the
  verdict at the probability you gave.
- `primary_finding` (object) -- `detector` (e.g. "D1"), `name`, `one_line`: the
  finding in one sentence naming the specific part, date or amount.
- `detectors` (array of 8) -- `detector`, `name`, `fired` (bool), `confidence`
  (int 0-100), `evidence` (the exact quoted document line or photograph), and
  `reason_not_fired` (why it stayed quiet, if it did).
- `photo_observations` (array, one per photograph supplied) -- `file`, `classes`
  (array), `damage_share` (e.g. "10%"), `severity` (`minor`|`light`|`moderate`|
  `heavy`|`severe`|`unclassifiable`), `description` (one line, what is visible).
- `part_support` (object) -- `supported`, `unsupported`, `unassessable` arrays.
- `cost_check` (object) -- `declared_estimate`, `quoted_total`,
  `market_expectation`, `ratio_quote_over_market`, `method` (one line: what you
  priced and at which band). Report the arithmetic even when D2 and D4 are quiet.
- `coverage_check` (object) -- `loss_date`, `policy_period`, `in_force` (bool),
  `note`.
- `counter_evidence` (array of strings) -- anything arguing against your own
  finding, including evidence you had to discount.
- `recommended_action` (string) -- one of: `settle without referral`,
  `request paint-meter reading`, `pull full claims history`,
  `request original file`, `contact named witness`,
  `refer repairer to compliance`, `coverage referral`, `escalate to SIU`.
- `next_steps` (array of strings) -- concrete things a human should do.
- `analyst_rationale` (string) -- the referral note, 3-5 paragraphs of plain
  prose, no bullet points. For each finding: what the document or photo actually
  says, the rule it breaks, why that rule exists, and the innocent explanation you
  considered and rejected. If you found nothing, say so plainly and say why. This
  is the field a claims manager actually reads, so write it for them rather than
  for the schema.
"""

ANTI_LEAK = """\
The pack has been anonymised: folder names and case references were deliberately
randomised and mean nothing. If you start reasoning about whether a claim "looks
like" it was meant to be genuine, stop and re-derive the answer from the evidence.
There is no signal in the labels, only in the evidence.
"""


def _fmt_rows(mapping: dict) -> str:
    return "\n".join(f"| {k} | {v} |" for k, v in mapping.items())


def _loss_type_rows() -> str:
    return "\n".join(
        f"| {k} | {', '.join(v) if v else '_none: impact damage is incompatible_'} |"
        for k, v in LOSS_TYPE_COMPATIBILITY.items()
    )


def _part_rows(price_book: dict) -> str:
    """One merged table: the classes a part accepts (D1) and its rates (D4)."""
    rows = ["| part | accepts these classes | category | rate lines (SGD ex GST) |",
            "|---|---|---|---|"]
    for part, spec in price_book["parts"].items():
        rates = "; ".join(f"{k} {v:,}" for k, v in spec["rates"].items())
        rows.append(f"| {part} | {', '.join(spec['expected_damage_classes'])} "
                    f"| {spec['category']} | {rates} |")
    return "\n".join(rows)


def _category_rows(price_book: dict) -> str:
    return "\n".join(f"- **{k}** -- {v}"
                     for k, v in price_book["category_upcharge_watchlist"].items())


def load_reference(reference_dir: Path = REFERENCE) -> tuple[dict, list[dict]]:
    """Return (price_book, prior_claims).

    Only the `prior_claims` array is loaded, never its `meta` block, which carries
    build-time commentary about individual cases.
    """
    price_book = json.loads((reference_dir / "price_book.json").read_text(encoding="utf-8"))
    prior_claims = json.loads(
        (reference_dir / "prior_claims.json").read_text(encoding="utf-8")
    )["prior_claims"]
    return price_book, prior_claims


def build_system_prompt(reference_dir: Path = REFERENCE) -> str:
    """Assemble the full system prompt, reference tables included."""
    price_book, prior_claims = load_reference(reference_dir)
    meta = price_book["meta"]
    total = price_book["expected_total"]

    calibration = "\n".join([
        "# Reference data",
        "",
        "## Severity calibration -- observed band from the share of each frame damaged,",
        "aggregated across the pack",
        "",
        "| band | share of frame damaged |",
        "|---|---|",
        _fmt_rows(meta["severity_by_observed_damage_pct"]),
        "",
        "| declared severity on the form | multiplier |",
        "|---|---|",
        _fmt_rows(meta["severity_multiplier"]),
        "",
        f"Consumables/labour overhead per job: {total['consumables_overhead']:,}.",
        f"Genuine-claim tolerance: declared estimate up to {total['tolerance_ratio_for_clean_case']}x "
        f"market expectation is unremarkable. Invoice escalation threshold: "
        f"{total['fraud_escalation_ratio_for_invoice_detector']}x. D6 fires at two thirds or more "
        "part overlap with a prior claim.",
        "",
        "## Compatibility -- declared nature of loss admits these damage classes",
        "",
        "| declared nature of loss | compatible classes |",
        "|---|---|",
        _loss_type_rows(),
        "",
        "## Parts -- accepted damage classes (use for D1) and market rates (use for D4)",
        "",
        _part_rows(price_book),
        "",
        "## Quotation categories that must be justified by the observed damage",
        "",
        _category_rows(price_book),
        "",
    ])

    history = json.dumps(prior_claims, indent=2) if prior_claims else "(empty)"

    sections = [
        ROLE,
        INPUTS,
        EVIDENCE_RULES.format(classes=", ".join(f"`{c}`" for c in DAMAGE_CLASSES)),
        DETECTORS,
        RECIPE,
        calibration,
        DISCIPLINE,
        OUTPUT,
        ANTI_LEAK,
        "# Prior claims history (portfolio lookup for D6)\n\n"
        "Search by vehicle registration or chassis number. Most vehicles have no\n"
        "history, which is the correct negative and must not be read as suspicious.\n\n"
        "```json\n" + history + "\n```",
    ]
    return "\n\n".join(sections)


def build_user_preamble(case_reference: str, documents: list[str], n_photos: int) -> str:
    """The per-case user turn that precedes the document text and the images."""
    listing = "\n".join(f"- {name}" for name in documents) or "- (none)"
    return f"""\
Assess this claim pack. Case reference: {case_reference}

Documents supplied:
{listing}

Damage photographs supplied: {n_photos}

The document text follows, then the photographs in filename order. Work the eight
detectors in order, classify every photograph before reasoning about part support,
and reply with the single JSON object specified in your instructions."""




import argparse
import base64
import io
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from urllib.parse import urlparse

import pypdf

try:
    import pypdfium2 as pdfium
except ImportError:  # only --doc-mode image needs this
    pdfium = None


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_env(ROOT / ".env")

# Convenience: Map GEMINI_API_KEY to FRAUD_API_KEY if not explicitly set
if "FRAUD_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["FRAUD_API_KEY"] = os.environ["GEMINI_API_KEY"]

DATASET = ROOT / "cases"
OUT_MD = ROOT / "FRAUD_DETECTION_RESULTS.md"
RAW_DIR = ROOT / "fraud_results_raw"
LOG_PATH = ROOT / "fraud_detection_run.log"

PHOTO_CLASSES = ", ".join(f"`{c}`" for c in DAMAGE_CLASSES)

BASE_URL = os.environ.get("FRAUD_API_BASE", "http://127.0.0.1:8080/v1")

# Measured against this model on this server, not guessed. Used only to choose a
# max_tokens that fits the window; the server is the authority on the real count
# and its usage figures are logged for every call.
CHARS_PER_TOKEN = 3.8
TOKENS_PER_IMAGE = 312

# Used only when the endpoint does not report its own context window, which is
# the normal case for a hosted API. It caps max_tokens so an oversized request is
# visibly clamped rather than silently rejected.
DEFAULT_N_CTX = 128_000

FOOTER_RE = re.compile(r"^(Case\s+\S+\s+\|.*Synthetic dataset.*|Page\s+\d+\s*)$")
BOILERPLATE_RE = re.compile(
    r"(Terms and conditions.*?stated\.|FOR OFFICE USE ONLY.*?claims system\.)",
    re.DOTALL,
)
REQUIRED_FIELDS = (
    "verdict", "fraud_probability", "confidence", "primary_finding",
    "detectors", "photo_observations", "analyst_rationale",
)


# =====================================================================
# Logging
# =====================================================================

class Log:
    """Timestamped stdout logger, optionally tee'd to a file."""

    def __init__(self, path: Path | None = None) -> None:
        self.handle = path.open("w", encoding="utf-8") if path else None

    def __call__(self, message: str = "", *, indent: int = 0) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {"  " * indent}{message}" if message else ""
        print(line, flush=True)
        if self.handle:
            self.handle.write(line + "\n")
            self.handle.flush()

    def rule(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def kv(self, key: str, value, *, indent: int = 2) -> None:
        self(f"{key}: {value}", indent=indent)

    def write_file(self, path: Path, text: str, *, what: str = "") -> None:
        """Announce, write, then read back and confirm -- writes are never silent."""
        payload = text.encode("utf-8")
        self(f"WRITING {what or path.name} -> {path} ({len(payload):,} bytes)", indent=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        check = path.read_bytes()
        status = "OK" if check == payload else "MISMATCH"
        self(f"WROTE   {path.name} verified {len(check):,} bytes [{status}]", indent=1)

    def close(self) -> None:
        if self.handle:
            self.handle.close()


# =====================================================================
# HTTP
# =====================================================================

def api_headers(args) -> dict:
    """Auth is the only header difference between a local server and a hosted API."""
    headers = {"Content-Type": "application/json"}
    key = getattr(args, "api_key", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def get_json(url: str, timeout: int = 20, args=None) -> dict:
    request = urllib.request.Request(url, headers=api_headers(args) if args else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class RateLimiter:
    """Sliding-window rate limiter ensuring calls do not exceed rpm requests per minute
    and enforcing an even inter-request pacing to avoid burst rate-limit errors."""

    def __init__(self, rpm: int = 5) -> None:
        self.rpm = rpm
        self.history: list[float] = []
        self.last_call: float = 0.0

    def wait(self, log: Log | None = None) -> None:
        if not self.rpm or self.rpm <= 0:
            return

        # Enforce minimum spacing between calls (e.g. 12s for 5 RPM)
        min_interval = 60.0 / self.rpm
        since_last = time.time() - self.last_call
        if self.last_call > 0 and since_last < min_interval:
            pace_sleep = min_interval - since_last
            if log:
                log(f"Rate limit pacing ({self.rpm} RPM): waiting {pace_sleep:.1f}s ...", indent=2)
            time.sleep(pace_sleep)

        window = 60.0
        while True:
            now = time.time()
            self.history = [t for t in self.history if now - t < window]
            if len(self.history) < self.rpm:
                now = time.time()
                self.history.append(now)
                self.last_call = now
                return
            sleep_time = window - (now - self.history[0]) + 0.5
            if sleep_time > 0:
                if log:
                    log(f"Rate limit sliding window ({self.rpm} RPM cap): waiting {sleep_time:.1f}s ...", indent=2)
                time.sleep(sleep_time)


class PhotoCache:
    """Computes SHA-256 hashes of incoming vehicle photos and caches Stage 1
    damage classifications to avoid redundant vision model calls on duplicate
    submissions or cross-claim identical photos."""

    def __init__(self, cache_dir: Path | None = None, gcs_bucket: str | None = None) -> None:
        self.cache_dir = cache_dir or (ROOT / "photo_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.gcs_bucket = gcs_bucket or os.environ.get("GCS_BUCKET")
        self._gcs_client = None

    @staticmethod
    def hash_blob(blob: bytes) -> str:
        return hashlib.sha256(blob).hexdigest()

    def _get_gcs_blob(self, sha256: str):
        if not self.gcs_bucket:
            return None
        if self._gcs_client is None:
            try:
                from google.cloud import storage
                self._gcs_client = storage.Client()
            except Exception:
                return None
        try:
            bucket = self._gcs_client.bucket(self.gcs_bucket)
            return bucket.blob(f"cache/photos/{sha256}.json")
        except Exception:
            return None

    def get(self, sha256: str) -> dict | None:
        local_file = self.cache_dir / f"{sha256}.json"
        if local_file.exists():
            try:
                with open(local_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Check Cloud Storage bucket cache
        blob = self._get_gcs_blob(sha256)
        if blob and blob.exists():
            try:
                data = json.loads(blob.download_as_text(encoding="utf-8"))
                with open(local_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                return data
            except Exception:
                pass
        return None

    def put(self, sha256: str, observation: dict, case_ref: str = "") -> None:
        data = dict(observation)
        data["sha256"] = sha256
        data["first_seen_case"] = data.get("first_seen_case") or case_ref
        data["cached_at"] = datetime.now().isoformat()

        # Save locally
        local_file = self.cache_dir / f"{sha256}.json"
        try:
            with open(local_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

        # Save to GCS bucket cache
        blob = self._get_gcs_blob(sha256)
        if blob:
            try:
                blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
            except Exception:
                pass


def post_json(url: str, payload: dict, timeout: int, args=None, log: Log | None = None) -> dict:
    limiter = getattr(args, "rate_limiter", None) if args else None
    max_retries = 4
    for attempt in range(1, max_retries + 1):
        if limiter:
            limiter.wait(log=log)
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers=api_headers(args) if args else {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", "replace")
            except Exception:
                pass
            is_quota_exhausted = any(k in err_body.lower() for k in ["quota", "exceeded", "resource_exhausted"])
            if is_quota_exhausted:
                if log:
                    log(f"HTTP 429 Quota exhausted for model. Failing over immediately to backup model.", indent=2)
                raise exc

            if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                rpm_val = getattr(args, "rpm", 5) or 5
                retry_after = max(60.0 / rpm_val, 15.0 * attempt)
                header_val = exc.headers.get("Retry-After")
                if header_val and header_val.isdigit():
                    retry_after = float(header_val) + 1.0
                if log:
                    reason = "rate limit (429)" if exc.code == 429 else f"server status {exc.code}"
                    log(f"HTTP {exc.code} ({reason}). Backing off for {retry_after:.1f}s (retry {attempt}/{max_retries}) ...", indent=2)
                time.sleep(retry_after)
                continue
            raise


def api_root(base_url: str) -> str:
    """Server root, for endpoints that live outside the version path.

    llama.cpp serves `/health` and `/props` at the root while `/v1/chat/completions`
    sits under the version path, so the base URL has to be trimmed for those.
    """
    trimmed = base_url.rstrip("/")
    # Strip repeatedly so a pasted full endpoint such as
    # ".../v1/chat/completions" reduces to the server root, not to ".../v1".
    suffixes = ("/chat/completions", "/completions", "/v1")
    stripping = True
    while stripping:
        stripping = False
        for suffix in suffixes:
            if trimmed.endswith(suffix):
                trimmed = trimmed[: -len(suffix)]
                stripping = True
                break
    return trimmed


def resolve_flavor(args) -> str:
    """Pick the request dialect. llama.cpp takes two extra fields that a hosted
    OpenAI endpoint rejects, so guessing wrong is a hard 400 rather than a
    fallback. Localhost means llama.cpp unless told otherwise."""
    if args.flavor != "auto":
        return args.flavor
    host = urlparse(args.base_url).hostname or ""
    return "llamacpp" if host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"} else "openai"


class ServerError(RuntimeError):
    pass


# =====================================================================
# Document and image handling
# =====================================================================

def pdf_text(path: Path, aggressive: bool) -> str:
    """Extract PDF text, dropping the repeated page furniture."""
    raw = "\n".join((page.extract_text() or "") for page in pypdf.PdfReader(str(path)).pages)
    lines = [line.rstrip() for line in raw.splitlines()
             if not FOOTER_RE.match(line.strip())]
    if aggressive:
        text = "\n".join(lines)
        text = BOILERPLATE_RE.sub("[standard terms and conditions / office-use block omitted]",
                                  text)
        lines = text.splitlines()
    lines = [re.sub(r"[ \t]{2,}", "  ", line) for line in lines]
    out: list[str] = []
    blanks = 0
    for line in lines:
        if line.strip():
            blanks = 0
        else:
            blanks += 1
            if blanks > 1:
                continue
        out.append(line)
    return "\n".join(out).strip()


def pdf_page_images(path: Path, scale: float) -> list[bytes]:
    """Rasterise each PDF page to a JPEG, for --doc-mode image."""
    if pdfium is None:
        raise ServerError(
            "--doc-mode image needs pypdfium2, which is not installed.\n"
            "  pip install pypdfium2\n"
            "  Or run in the default text mode, which needs no extra dependency."
        )
    document = pdfium.PdfDocument(str(path))
    blobs = []
    for page in document:
        image = page.render(scale=scale).to_pil().convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
        blobs.append(buffer.getvalue())
    return blobs


def image_part(blob: bytes, mime: str = "image/jpeg") -> dict:
    encoded = base64.b64encode(blob).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


class CasePack:
    """Everything one case contributes to the request, plus its accounting."""

    def __init__(self, case_dir: Path, doc_mode: str, render_scale: float) -> None:
        self.dir = case_dir
        self.reference = case_dir.name
        self.parts: list[dict] = []
        self.doc_names: list[str] = []
        self.photos: list[str] = []
        self.photo_blobs: list[tuple[str, bytes]] = []
        self.chars = 0
        self.images = 0

        pdfs = sorted(case_dir.glob("*.pdf"))
        evidence_dir = case_dir / "evidence"
        photos = []
        if evidence_dir.is_dir():
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                photos.extend(evidence_dir.glob(pattern))
            photos = sorted(set(photos))

        if doc_mode == "image":
            for pdf in pdfs:
                blobs = pdf_page_images(pdf, render_scale)
                for number, blob in enumerate(blobs, start=1):
                    self.parts.append({"type": "text", "text":
                                       f"[{pdf.name} page {number} of {len(blobs)} follows]"})
                    self.parts.append(image_part(blob))
                    self.images += 1
                self.doc_names.append(f"{pdf.name} ({len(blobs)} page images)")
        else:
            for pdf in pdfs:
                text = pdf_text(pdf, aggressive=False)
                self.chars += len(text)
                self.parts.append({"type": "text", "text":
                                   f"--- BEGIN {pdf.name} ---\n{text}\n--- END {pdf.name} ---"})
                self.doc_names.append(pdf.name)

        for photo in photos:
            blob = photo.read_bytes()
            self.parts.append({"type": "text", "text":
                               f"[{photo.name} follows]"})
            self.parts.append(image_part(blob))
            self.photos.append(photo.name)
            self.photo_blobs.append((photo.name, blob))
            self.images += 1

    def stage2_parts(self, observations: list[dict]) -> list[dict]:
        """Document text plus the stage-1 photo readings, and no raw images.

        Stage 2 must not see the photographs again. Left with the pixels it
        re-classifies them inconsistently with its own stage-1 reading and then
        reasons from the worse of the two; the observations are the record of
        what the images showed.
        """
        parts = [p for p in self.parts if p.get("type") == "text"
                 and not p["text"].startswith("[") and not p["text"].endswith("follows]")]
        parts = [p for p in parts if p["text"] not in
                 {f"[{name} follows]" for name in self.photos}]
        if observations:
            parts.append({"type": "text", "text": render_observations(observations)})
        return parts

    def estimate_tokens(self, system_chars: int) -> int:
        return int((system_chars + self.chars) / CHARS_PER_TOKEN) + self.images * TOKENS_PER_IMAGE

    def compress(self) -> None:
        """Second-pass squeeze: drop boilerplate blocks from text parts."""
        rebuilt: list[dict] = []
        for part in self.parts:
            if part.get("type") == "text" and part["text"].startswith("--- BEGIN "):
                name = part["text"].split("--- BEGIN ")[1].split(" ---")[0]
                body = part["text"].split("\n", 1)[1].rsplit(f"\n--- END {name} ---", 1)[0]
                squeezed = pdf_text_text(body)
                self.chars += len(squeezed) - len(body)
                rebuilt.append({"type": "text", "text":
                                f"--- BEGIN {name} ---\n{squeezed}\n--- END {name} ---"})
            else:
                rebuilt.append(part)
        self.parts = rebuilt


def pdf_text_text(body: str) -> str:
    return BOILERPLATE_RE.sub("[standard terms and conditions / office-use block omitted]", body)


# =====================================================================
# LLM call
# =====================================================================

def strip_json_fence(raw: str) -> str:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in the model response")
    return text[start:end + 1]


PHOTO_STAGE_SYSTEM = """\
You are the photographic evidence reader on a motor insurance claim. You do not
score fraud and you do not decide anything. You only report what each
photograph shows, and you are being read by another system that will make the
call, so accuracy matters far more than confidence.

Every photograph here is a tight close-up crop of a damaged area. None of them
shows the whole vehicle, so you CANNOT identify which panel or part of the car
is in frame, and you must not try. A crop of a bumper and a crop of a mirror
can look almost identical at this scale, so any part name you produce would be
a guess dressed up as an observation. Report the damage, never the panel.

The complete and only list of damage classes is:
{classes}

For EACH photograph return an object with:
  file         the filename exactly as given
  classes      the classes genuinely visible, comma separated from the list
               above, or an empty list if none is clearly visible
  severity     minor | moderate | severe, from how much of the frame is damaged
  share        rough percentage of the frame that is damaged
  description  one line on what the damage physically looks like

Rules:
- A class counts only if you can see it. Never infer a class from context.
- If a photograph is too small, dark, blurred or ambiguous to read, return an
  empty classes list and say why in the description. An unreadable photograph is
  itself a finding.
- Never merge photographs. One object per photograph, same filenames, same
  order, no additions and no omissions.

Return ONLY a JSON array of those objects. No commentary, no explanation.
"""


def render_observations(observations: list[dict]) -> str:
    """The stage-1 photo readings, formatted for stage 2 as the visual record."""
    if not observations:
        return ("--- PHOTOGRAPHIC EVIDENCE ---\n"
                "No photographs were attached to this claim.\n"
                "--- END PHOTOGRAPHIC EVIDENCE ---")
    lines = ["--- PHOTOGRAPHIC EVIDENCE (read from the images; no panels identified) ---"]
    for item in observations:
        classes = item.get("classes") or []
        shown = ", ".join(str(c) for c in classes) if classes else "none clearly visible"
        lines.append(f"{item.get('file', '?')}: classes [{shown}]")
        if item.get("severity"):
            lines.append(f"  severity: {item['severity']}"
                         + (f" | share of frame: {item['share']}" if item.get("share") else ""))
        if item.get("description"):
            lines.append(f"  looks like: {one_line(item['description'], 220)}")
    lines.append("--- END PHOTOGRAPHIC EVIDENCE ---")
    return "\n".join(lines)


def parse_observations(content: str, expected: list[str]) -> list[dict]:
    """Parse the stage-1 array and refuse to proceed on a short or misfiled read."""
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in the photo-reading response")
    items = json.loads(text[start:end + 1])
    if not isinstance(items, list):
        raise ValueError("photo reading was not a JSON array")

    if expected:
        got = [str(i.get("file", "")).strip() for i in items]
        missing = [name for name in expected if name not in got]
        if missing:
            raise ValueError(f"photo reading omitted {missing}; got {got}")
    for item in items:
        classes = item.get("classes")
        if isinstance(classes, str):
            classes = [c.strip() for c in classes.split(",") if c.strip()]
        item["classes"] = classes if isinstance(classes, list) else []
    return items


def call_model(system: str, user_parts, args, max_tokens: int, thinking: bool,
               label: str, log: Log, model_override: str | None = None) -> tuple[str, dict]:
    model_name = model_override or getattr(args, "model", "gemini-3.8-flash")
    payload = {
        "model": model_name,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_parts}],
        "temperature": args.temperature,
        "max_tokens": max_tokens,
    }
    if resolve_flavor(args) == "llamacpp":
        payload["cache_prompt"] = True
        # Gemma 4 reasons in a separate channel before it answers. On this model
        # reasoning is what makes a photograph readable, but on the detector task
        # it does not terminate, so stage 2 runs with it off.
        payload["chat_template_kwargs"] = {"enable_thinking": thinking}
    elif ("gemini-3.7" in model_name.lower() or "gemini-3.8" in model_name.lower()) and not thinking:
        payload["reasoning_effort"] = "none"

    log(f"POST {args.base_url}/chat/completions  [{label}]", indent=1)
    log(f"  dialect        : {resolve_flavor(args)}", indent=1)
    log(f"  model          : {model_name}", indent=1)
    log(f"  max_tokens     : {max_tokens}", indent=1)
    log(f"  temperature    : {args.temperature}", indent=1)
    if resolve_flavor(args) == "llamacpp":
        log(f"  thinking channel: {'on' if thinking else 'off'}", indent=1)

    started = time.time()
    body = post_json(f"{args.base_url}/chat/completions", payload, args.timeout, args, log=log)
    elapsed = time.time() - started

    message = body["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = body.get("usage", {})
    finish = body["choices"][0].get("finish_reason")

    log(f"  response in {elapsed:.1f}s | finish_reason={finish}", indent=1)
    log(f"  usage: prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')}", indent=1)
    if reasoning:
        log(f"  model reasoning: {len(reasoning):,} chars (not shown in full)", indent=1)
    log("-" * 74, indent=1)
    log(f"RAW LLM RESPONSE ({label}) [{model_name}]:", indent=1)
    for line in content.splitlines():
        log(line, indent=2)
    log("-" * 74, indent=1)
    if not content:
        log("EMPTY CONTENT -- budget spent on reasoning with nothing returned", indent=1)
    return content, {"usage": usage, "finish": finish, "elapsed": elapsed,
                     "reasoning_chars": len(reasoning), "model": model_name}


def call_model_with_fallback(system: str, user_parts, args, max_tokens: int, thinking: bool,
                            label: str, log: Log, chain: list[str]) -> tuple[str, dict]:
    last_exc: Exception | None = None
    for idx, model_name in enumerate(chain):
        if idx > 0:
            log(f"FALLBACK -> switching to backup model '{model_name}' [{idx + 1}/{len(chain)}] for {label}", indent=1)
        try:
            return call_model(system, user_parts, args, max_tokens, thinking, label, log, model_override=model_name)
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in {429, 500, 502, 503, 504} and idx < len(chain) - 1:
                next_model = chain[idx + 1]
                log(f"WARNING: Model '{model_name}' hit HTTP {exc.code} (quota/rate limit). Falling back to backup '{next_model}' ...", indent=1)
                continue
            raise
        except (ServerError, urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if idx < len(chain) - 1:
                next_model = chain[idx + 1]
                log(f"WARNING: Model '{model_name}' failed with {exc!r}. Falling back to backup '{next_model}' ...", indent=1)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"All models in chain {chain} failed")


def classify_photos(pack: CasePack, classes: str, args, log: Log) -> list[dict]:
    """Stage 1: read the damage class off every photograph.

    Includes SHA-256 deduplication & caching:
    - Computes SHA-256 hash of each photo blob.
    - If cached, reuses the damage reading without making a vision API call.
    - Only sends uncached photos to the vision model, then writes them to cache.
    """
    if not pack.photo_blobs:
        log("no photographs; skipping the photo-reading stage", indent=1)
        return []

    cache = getattr(args, "photo_cache", None) or PhotoCache()
    final_observations: dict[str, dict] = {}
    missing_blobs: list[tuple[str, bytes, str]] = []

    for name, blob in pack.photo_blobs:
        sha256 = PhotoCache.hash_blob(blob)
        cached = cache.get(sha256)
        if cached:
            obs = {
                "file": name,
                "classes": cached.get("classes", []),
                "severity": cached.get("severity", "minor"),
                "share": cached.get("share", ""),
                "description": cached.get("description", ""),
                "_cached": True,
                "_sha256": sha256,
            }
            cross_claim = f" [first observed in {cached.get('first_seen_case')}]" if cached.get("first_seen_case") and cached.get("first_seen_case") != pack.reference else ""
            log(f"  {name} (SHA-256: {sha256[:10]}...) -> CACHE HIT! Reusing damage reading{cross_claim} (0 model calls)", indent=1)
            final_observations[name] = obs
        else:
            missing_blobs.append((name, blob, sha256))

    if not missing_blobs:
        log(f"STAGE 1: All {len(pack.photo_blobs)} photo(s) retrieved from cache! Vision model calls avoided.", indent=1)
        return [final_observations[name] for name, _ in pack.photo_blobs]

    log(f"STAGE 1 photo reading: {len(missing_blobs)}/{len(pack.photo_blobs)} new image(s) to process ({len(final_observations)} cached)", indent=1)

    parts: list[dict] = []
    missing_names = []
    for name, blob, _ in missing_blobs:
        parts.append({"type": "text", "text": name})
        parts.append(image_part(blob))
        missing_names.append(name)

    system = PHOTO_STAGE_SYSTEM.format(classes=classes)
    ladder = [
        (args.photo_max_tokens, True),
        (args.photo_max_tokens * 2, True),
        (args.photo_max_tokens, False),
    ]
    failure: Exception | None = None
    for attempt, (budget, thinking) in enumerate(ladder, start=1):
        if attempt > 1:
            log(f"stage 1 retry {attempt}/{len(ladder)}: "
                f"{budget} tokens, reasoning {'on' if thinking else 'off'}", indent=1)
        content, _meta = call_model_with_fallback(system, parts, args, budget, thinking,
                                                 "stage1 photos", log, chain=args.vision_chain)
        try:
            observations = parse_observations(content, missing_names)
        except (ValueError, json.JSONDecodeError) as exc:
            failure = exc
            log(f"stage 1 attempt {attempt} unusable: {exc}", indent=1)
            continue
        if attempt > 1:
            log(f"stage 1 recovered on attempt {attempt}", indent=1)
        
        # Save newly read observations into cache
        for item in observations:
            name = item.get("file")
            matching_hash = next((h for n, _, h in missing_blobs if n == name), None)
            if matching_hash:
                cache.put(matching_hash, item, case_ref=pack.reference)
            final_observations[name] = item
            log(f"  {name}: classes={item.get('classes')} "
                f"severity={item.get('severity')} [cached with SHA-256: {matching_hash[:10] if matching_hash else '?'}]", indent=1)
        
        return [final_observations[name] for name, _ in pack.photo_blobs]

    raise failure or ValueError("photo reading failed on every attempt")


def analyse(pack: CasePack, observations: list[dict], system_prompt: str,
            args, n_ctx: int, log: Log) -> dict:
    """Stage 2: the detector run, on documents plus the stage-1 photo readings."""
    stage2 = pack.stage2_parts(observations)
    chars = sum(len(p["text"]) for p in stage2)
    budget = n_ctx - int((len(system_prompt) + chars) / CHARS_PER_TOKEN) - 96
    max_tokens = max(400, min(args.max_output_tokens, budget))
    if budget < args.max_output_tokens:
        log(f"WARNING context budget clamped: {budget} tokens available for output, "
            f"requested {args.max_output_tokens}", indent=1)
    if max_tokens < 900:
        log("WARNING output budget under 900 tokens, the rationale may be truncated",
            indent=1)

    log(f"STAGE 2 detector run", indent=1)
    log(f"  prompt estimate : {int((len(system_prompt) + chars) / CHARS_PER_TOKEN):,} "
        f"tokens ({len(system_prompt):,} prompt chars + {chars:,} doc/observation chars)",
        indent=1)
    content, meta = call_model_with_fallback(system_prompt, stage2, args, max_tokens,
                                            False, "stage2 detectors", log, chain=args.text_chain)

    try:
        parsed = json.loads(strip_json_fence(content))
    except (ValueError, json.JSONDecodeError) as exc:
        # Stage 2 answers without reasoning, so truncation is rare, but a claim
        # must never die on one malformed generation.
        log(f"stage 2 verdict unusable ({exc}); retrying once with double budget",
            indent=1)
        content, meta = call_model_with_fallback(system_prompt, stage2, args, max_tokens * 2,
                                                False, "stage2 detectors retry", log, chain=args.text_chain)

        parsed = json.loads(strip_json_fence(content))
    parsed["_photo_observations"] = observations
    parsed["_usage"] = meta["usage"]
    parsed["_finish_reason"] = meta["finish"]
    parsed["_elapsed_s"] = round(meta["elapsed"], 1)
    parsed["_reasoning_chars"] = meta["reasoning_chars"]

    missing = [f for f in REQUIRED_FIELDS if f not in parsed]
    if missing:
        log(f"WARNING verdict JSON missing fields: {missing}", indent=1)
    if meta["finish"] == "length":
        log("WARNING response hit the token limit, JSON may be truncated", indent=1)
    return parsed


# =====================================================================
# Markdown report
# =====================================================================

def cell(value) -> str:
    return str(value if value is not None else "--").replace("|", r"\|").replace("\n", " ").strip()


def one_line(value, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "\u2026"


def bar(probability) -> str:
    try:
        value = max(0, min(100, int(probability)))
    except (TypeError, ValueError):
        return ""
    filled = round(value / 5)
    return "#" * filled + "." * (20 - filled)


def fired_detectors(parsed: dict) -> list[dict]:
    return [d for d in parsed.get("detectors", []) if d.get("fired")]


def verdict_section(case_ref: str, parsed: dict) -> str:
    probability = parsed.get("fraud_probability", 0)
    verdict = parsed.get("verdict", "unknown")
    primary = parsed.get("primary_finding") or {}
    lines = [
        f"### {case_ref} -- {str(verdict).upper()} ({probability}%)",
        "",
        f"`{bar(probability)}` **{probability}%** likelihood of fraud "
        f"| confidence: **{parsed.get('confidence', '--')}**",
        "",
    ]
    primary_line = primary.get("one_line") or "no supportable finding"
    primary_name = " / ".join(x for x in (primary.get("detector"), primary.get("name")) if x)
    lines += [
        f"**Primary finding** ({primary_name or 'none'}): {primary_line}",
        "",
        f"**Recommended action:** `{parsed.get('recommended_action', '--')}`",
        "",
        "**Detectors**",
        "",
        "| detector | technique | fired | conf | evidence / why it stayed quiet |",
        "|---|---|:-:|---:|---|",
    ]
    for detector in parsed.get("detectors", []):
        note = detector.get("evidence") if detector.get("fired") else detector.get("reason_not_fired")
        lines.append(
            f"| {cell(detector.get('detector'))} | {cell(detector.get('name'))} "
            f"| {'YES' if detector.get('fired') else 'no'} "
            f"| {cell(detector.get('confidence'))} | {cell(one_line(note, 220))} |"
        )
    if not parsed.get("detectors"):
        lines.append("| -- | -- | -- | -- | the model returned no detector array |")

    observations = parsed.get("photo_observations") or []
    lines += ["", "**Photograph observations**", "",
              "| file | classes observed | share of frame | severity | description |",
              "|---|---|---|---|---|"]
    if observations:
        for photo in observations:
            classes = ", ".join(photo.get("classes") or []) or "--"
            lines.append(
                f"| {cell(photo.get('file'))} | {cell(classes)} "
                f"| {cell(photo.get('damage_share'))} | {cell(photo.get('severity'))} "
                f"| {cell(one_line(photo.get('description'), 150))} |"
            )
    else:
        lines.append("| -- | -- | -- | -- | none returned |")

    support = parsed.get("part_support") or {}
    cost = parsed.get("cost_check") or {}
    coverage = parsed.get("coverage_check") or {}

    def names(key: str) -> str:
        values = support.get(key) or []
        rendered = []
        for v in values:
            if isinstance(v, dict):
                part = v.get("part") or str(v)
                cls = v.get("supporting_class")
                rendered.append(f"{part} ({cls})" if cls else str(part))
            else:
                rendered.append(str(v))
        return ", ".join(rendered) if rendered else "none"

    lines += [
        "",
        "**Part support** -- supported: "
        + cell(names("supported"))
        + " | unsupported: " + cell(names("unsupported"))
        + " | unassessable: " + cell(names("unassessable")),
        "",
        "**Cost check** -- declared estimate: "
        + cell(cost.get("declared_estimate"))
        + " | quoted total: " + cell(cost.get("quoted_total"))
        + " | market expectation: " + cell(cost.get("market_expectation"))
        + " | ratio quote/market: " + cell(cost.get("ratio_quote_over_market")),
        "",
        f"**Coverage check** -- loss date {cell(coverage.get('loss_date'))} "
        f"| policy period {cell(coverage.get('policy_period'))} "
        f"| in force: {cell(coverage.get('in_force'))} "
        f"| {cell(one_line(coverage.get('note'), 200))}",
        "",
    ]

    counter = parsed.get("counter_evidence") or []
    lines.append("**Counter-evidence considered**")
    lines.append("")
    lines += [f"- {item}" for item in counter] or ["- none reported"]
    lines.append("")

    steps = parsed.get("next_steps") or []
    lines.append("**Next steps**")
    lines.append("")
    lines += [f"- {item}" for item in steps] or ["- none recommended"]
    lines += ["", "**Analyst rationale**", "", "> " +
              "\n> ".join(str(parsed.get("analyst_rationale", "")).splitlines()), ""]
    return "\n".join(lines)


def summary_row(case_ref: str, parsed: dict) -> str:
    primary = parsed.get("primary_finding") or {}
    label = primary.get("detector") or "--"
    return (f"| `{case_ref}` | **{cell(parsed.get('verdict'))}** "
            f"| {cell(parsed.get('fraud_probability'))}% "
            f"| {cell(parsed.get('confidence'))} | {cell(label)} "
            f"| {cell(one_line(primary.get('one_line'), 90))} "
            f"| {cell(parsed.get('recommended_action'))} |")


def load_truth() -> tuple[dict[str, dict], str]:
    """Ground truth by anonymised folder. Operator use only, never sent to the model."""
    path = ROOT / "ground_truth.json"
    if not path.is_file():
        return {}, ""
    return json.loads(path.read_text(encoding="utf-8")), str(path)



def score_section(results: list[tuple[str, dict]]) -> str:
    """Join predictions against ground truth. Operator use only."""
    by_folder, source = load_truth()
    if not by_folder:
        return ("## Ground-truth scoring\n\n"
                "No ground truth available: `ground_truth.json` was not found next to this script.")

    folder_to_original = {ref: rec.get("original_case_id", "--") for ref, rec in by_folder.items()}

    rows = ["## Ground-truth scoring (operator only -- not part of the model's view)", "",
            f"Scored against `{Path(source).name}`.", "",
            "| case | original | predicted | actual | fraud call | exact | prob | predicted primary | actual technique |",
            "|---|---|---|---|:-:|:-:|---:|---|---|"]
    fraud_correct = 0
    exact_correct = 0
    detector_hit = 0
    detector_total = 0
    scored = 0
    false_positive: list[str] = []
    false_negative: list[str] = []
    missed_technique: list[str] = []
    for case_ref, parsed in results:
        original = folder_to_original.get(case_ref, "--")
        record = by_folder.get(case_ref)
        if not record:
            continue
        scored += 1
        predicted = str(parsed.get("verdict", "")).lower()
        actual = record["verdict"]
        was_fraud = actual == "fraud"
        # The competition question is binary: is this fraud or not. `not_covered`
        # and `needs_investigation` both mean "not fraud", so they count as a correct
        # fraud call but not as an exact verdict match.
        says_fraud = predicted in {"fraud", "suspended"}
        fraud_ok = says_fraud == was_fraud
        exact_ok = predicted == actual
        fraud_correct += fraud_ok
        exact_correct += exact_ok
        if says_fraud and not was_fraud:
            false_positive.append(case_ref)
        if was_fraud and not says_fraud:
            false_negative.append(case_ref)
        predicted_primary = (parsed.get("primary_finding") or {}).get("detector") or "--"
        actual_technique = record.get("technique")
        fired = {str((d.get("detector") or "")).upper()
                 for d in (parsed.get("detectors") or []) if d.get("fired")}
        if actual_technique:
            detector_total += 1
            hit = actual_technique.upper() in fired
            detector_hit += hit
            if not hit:
                missed_technique.append(f"{case_ref}->{actual_technique}")
        rows.append(
            f"| `{case_ref}` | `{original}` | {cell(predicted)} | {cell(actual)} "
            f"| {'ok' if fraud_ok else 'MISS'} | {'ok' if exact_ok else '--'} "
            f"| {cell(parsed.get('fraud_probability'))}% "
            f"| {cell(predicted_primary)} | {cell(actual_technique)} "
            f"{cell(record.get('technique_name'))} |"
        )
    if scored:
        rows += ["",
                 f"**Fraud-call accuracy {fraud_correct}/{scored} "
                 f"({fraud_correct / scored:.0%})** -- the headline metric: is fraud or not.",
                 f"Exact verdict match {exact_correct}/{scored} ({exact_correct / scored:.0%}).",
                 f"Ground-truth technique among fired detectors "
                 f"{detector_hit}/{detector_total}.",
                 f"False positives (clean called fraud): {', '.join(false_positive) or 'none'}",
                 f"False negatives (fraud called clean): {', '.join(false_negative) or 'none'}",
                 f"Technique missed: {', '.join(missed_technique) or 'none'}", ""]
    else:
        rows += ["", "No ground truth available.", ""]
    return "\n".join(rows)


def build_report(results, args, n_ctx: int, system_chars: int, started: datetime,
                 elapsed: float, n_ctx_assumed: bool = False) -> str:
    header = [
        "# Motor Claims Fraud Detection Report",
        "",
        f"**Generated:** {started.strftime('%Y-%m-%d %H:%M:%S')}  |  **Duration:** {elapsed / 60:.1f} min",
        "",
        "## Summary",
        "",
        "| Case | Verdict | Fraud Probability | Confidence | Primary Detector | Finding | Recommended Action |",
        "|---|---|---:|---|---|---|---|",
    ]
    header += [summary_row(case_ref, parsed) for case_ref, parsed in results]
    header += ["", "## Detailed Case Analysis", ""]
    body = "\n\n".join(verdict_section(case_ref, parsed) for case_ref, parsed in results)
    report = "\n".join(header) + "\n" + body + "\n"
    if getattr(args, "score", False):
        report += "\n" + score_section(results) + "\n"
    return report


# =====================================================================
# Main
# =====================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=BASE_URL,
                        help="API base URL including the version path, e.g. "
                             "http://127.0.0.1:8080/v1 or https://host/v1 "
                             "(env FRAUD_API_BASE)")
    parser.add_argument("--api-key", default=os.environ.get("FRAUD_API_KEY", ""),
                        help="bearer token for hosted endpoints; unused by a local "
                             "server (env FRAUD_API_KEY)")
    default_vchain = os.environ.get("FRAUD_VISION_CHAIN", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash")
    default_tchain = os.environ.get("FRAUD_TEXT_CHAIN", "gemini-3.7-flash,gemini-3.5-flash-lite,gemini-2.5-flash")
    parser.add_argument("--model", default=os.environ.get("FRAUD_MODEL", "gemini-3.8-flash"),
                        help="fallback model name (env FRAUD_MODEL)")
    parser.add_argument("--vision-chain", default=default_vchain,
                        help="comma-separated fallback chain for stage 1 vision (env FRAUD_VISION_CHAIN)")
    parser.add_argument("--text-chain", default=default_tchain,
                        help="comma-separated fallback chain for stage 2 text (env FRAUD_TEXT_CHAIN)")
    parser.add_argument("--vision-model",
                        default=os.environ.get("FRAUD_VISION_MODEL", ""),
                        help="override primary vision model (env FRAUD_VISION_MODEL)")
    parser.add_argument("--text-model",
                        default=os.environ.get("FRAUD_TEXT_MODEL", ""),
                        help="override primary text model (env FRAUD_TEXT_MODEL)")
    parser.add_argument("--flavor", choices=("auto", "llamacpp", "openai"),
                        default=os.environ.get("FRAUD_API_FLAVOR", "auto"),
                        help="request dialect; auto means llamacpp for localhost "
                             "(env FRAUD_API_FLAVOR)")
    parser.add_argument("--dataset", type=Path, default=DATASET, help="directory of case folders")
    parser.add_argument("--out", type=Path, default=OUT_MD, help="markdown report path")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="per-case JSON sidecars")
    parser.add_argument("--log-file", type=Path, default=LOG_PATH, help="tee the debug log here")
    parser.add_argument("--case", action="append", default=[], metavar="CASE-NNN",
                        help="only this case folder (repeatable)")
    parser.add_argument("--doc-mode", choices=("text", "image"), default="text",
                        help="send PDF text, or rasterise PDF pages as images")
    parser.add_argument("--render-scale", type=float, default=2.0,
                        help="rasterisation scale for --doc-mode image")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--photo-max-tokens", type=int, default=4000,
                        help="stage-1 photo-reading budget; needs ~2.5k with "
                             "reasoning on, and is doubled automatically on a retry")
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="stage-2 detector budget; default 8192 tokens")
    parser.add_argument("--timeout", type=int, default=900, help="per-request seconds")
    parser.add_argument("--score", action="store_true",
                        help="append a ground-truth scoring section, read from "
                             "ground_truth.json")
    parser.add_argument("--rescore", action="store_true",
                        help="rebuild the markdown report from the raw JSON already in "
                             "--raw-dir without calling the model again")
    parser.add_argument("--rpm", "--requests-per-minute", type=int,
                        default=int(os.environ.get("FRAUD_RPM", os.environ.get("FRAUD_REQUESTS_PER_MINUTE", "5"))),
                        help="maximum requests per minute (env FRAUD_RPM or FRAUD_REQUESTS_PER_MINUTE, default 5)")
    parser.add_argument("--no-log-file", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.base_url = args.base_url.rstrip("/")

    # Build fallback chains
    args.vision_chain = [m.strip() for m in args.vision_chain.split(",") if m.strip()]
    args.text_chain = [m.strip() for m in args.text_chain.split(",") if m.strip()]
    if args.vision_model:
        if args.vision_model in args.vision_chain:
            args.vision_chain.remove(args.vision_model)
        args.vision_chain.insert(0, args.vision_model)
    if args.text_model:
        if args.text_model in args.text_chain:
            args.text_chain.remove(args.text_model)
        args.text_chain.insert(0, args.text_model)

    args.vision_model = args.vision_chain[0]
    args.text_model = args.text_chain[0]
    args.model_note = f"vision: {' -> '.join(args.vision_chain)} | text: {' -> '.join(args.text_chain)}"
    args.rate_limiter = RateLimiter(args.rpm)
    args.photo_cache = PhotoCache()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    log = Log(None if args.no_log_file else args.log_file)
    started = datetime.now()
    clock = time.time()

    log.rule("START")
    log.kv("script", Path(__file__).name)
    log.kv("dataset", args.dataset)
    log.kv("report", args.out)
    log.kv("doc mode", args.doc_mode)
    log.kv("vision chain", " -> ".join(args.vision_chain))
    log.kv("text chain", " -> ".join(args.text_chain))
    log.kv("rate limit", f"{args.rpm} RPM" if args.rpm > 0 else "unlimited")
    log.kv("log file", "disabled" if args.no_log_file else args.log_file)

    if args.doc_mode == "image" and pdfium is None:
        log("FATAL --doc-mode image needs pypdfium2, which is not installed.")
        log("  pip install pypdfium2")
        log("  Or use the default text mode, which has no extra dependency.")
        log.close()
        return 2

    log("")
    log("Loading reference data and assembling the system prompt ...")
    system_prompt = build_system_prompt(REFERENCE)
    log.kv("system prompt", f"{len(system_prompt):,} chars "
                            f"(~{int(len(system_prompt) / CHARS_PER_TOKEN):,} tokens)")

    if not args.dataset.is_dir():
        log(f"FATAL dataset directory not found: {args.dataset}")
        log.close()
        return 2

    # --rescore only reformats JSON that is already on disk, so it must not need
    # the model endpoint. Do it before the health check so a report can be
    # rebuilt with the server switched off.
    if args.rescore:
        saved = sorted(args.raw_dir.glob("*.json"))
        if not saved:
            log(f"FATAL no raw JSON in {args.raw_dir} to rescore from")
            log.close()
            return 2
        log("")
        log.kv("rescoring", f"{len(saved)} saved result(s) from {args.raw_dir}")
        results = [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in saved]
        report = build_report(results, args, DEFAULT_N_CTX, len(system_prompt), started,
                              time.time() - clock, n_ctx_assumed=True)
        log.rule("REPORT")
        log.write_file(args.out, report, what="markdown report (rescored)")
        log.close()
        return 0

    log("")
    log("Checking the model endpoint ...")
    # A hosted OpenAI-compatible API has no /health or /props. Probe what the
    # chosen dialect can answer, and fall back to documented defaults rather than
    # refusing to run against an endpoint that may be perfectly fine.
    n_ctx = DEFAULT_N_CTX
    vision = None
    slots = None
    status = "unverified"
    try:
        if resolve_flavor(args) == "llamacpp":
            root = api_root(args.base_url)
            health = get_json(f"{root}/health", timeout=15, args=args)
            props = get_json(f"{root}/props", timeout=30, args=args)
            n_ctx = props["default_generation_settings"]["n_ctx"]
            vision = props.get("modalities", {}).get("vision")
            slots = props.get("total_slots")
            status = health.get("status")
            args.model_note = Path(props.get("model_alias", "")).name or f"vision: {args.vision_model}, text: {args.text_model}"
        else:
            listing = get_json(f"{args.base_url}/models", timeout=20, args=args)
            names = [m.get("id") for m in listing.get("data", [])]
            status = "ok"

    except urllib.error.HTTPError as exc:
        log(f"FATAL {args.base_url} rejected the request: HTTP {exc.code}")
        log(f"  {exc.read().decode('utf-8', 'replace')[:400]}")
        if args.api_key or os.environ.get("FRAUD_API_KEY"):
            log("  An API key is set but may be wrong or expired.")
        log("  Check --base-url, --api-key and --model.")
        log.close()
        return 2
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        log(f"FATAL cannot reach {args.base_url}: {exc}")
        log("For a local llama.cpp server, start it first, for example:")
        log('  llama-server.exe -m "<model>.gguf" --mmproj "<mmproj>.gguf" '
            '-ngl 99 -c 32768 -np 1 -fa on -ub 2048 --cache-type-k q4_0 --cache-type-v q4_0')
        log("For a hosted model, pass --base-url, --api-key and --model.")
        log.close()
        return 2
    log.kv("endpoint", f"{args.base_url} ({resolve_flavor(args)})")
    log.kv("health", status)
    log.kv("model", args.model_note)
    log.kv("n_ctx", f"{n_ctx:,}" if vision is not None else f"{n_ctx:,} (assumed)")
    if vision is not None:
        log.kv("vision", vision)
    if slots is not None:
        log.kv("parallel slots", slots)

    case_dirs = sorted(p for p in args.dataset.iterdir() if p.is_dir())
    if args.case:
        wanted = {c.strip().upper() for c in args.case}
        case_dirs = [p for p in case_dirs if p.name.upper() in wanted]
        log.kv("filtered to", ", ".join(p.name for p in case_dirs))
    log.kv("cases found", len(case_dirs))

    results: list[tuple[str, dict]] = []
    failures: list[tuple[str, str]] = []

    for index, case_dir in enumerate(case_dirs, start=1):
        log.rule(f"CASE {index}/{len(case_dirs)}  {case_dir.name}")
        log.kv("path", case_dir)

        files = sorted(p.relative_to(case_dir).as_posix()
                       for p in case_dir.rglob("*") if p.is_file())
        log.kv("files", len(files))
        for name in files:
            size = (case_dir / name).stat().st_size
            log(f"{name}  ({size:,} bytes)", indent=1)

        pack = CasePack(case_dir, args.doc_mode, args.render_scale)
        log.kv("documents", ", ".join(pack.doc_names) or "none")
        log.kv("photographs", f"{len(pack.photos)} ({', '.join(pack.photos)})"
               if pack.photos else "none")
        log.kv("document chars", f"{pack.chars:,}")
        log.kv("images attached", pack.images)

        pack.parts.insert(0, {"type": "text", "text": build_user_preamble(
            case_dir.name, pack.doc_names, len(pack.photos))})

        estimate = pack.estimate_tokens(len(system_prompt))
        log.kv("prompt estimate", f"{estimate:,} tokens of {n_ctx:,}")
        if estimate + 900 > n_ctx:
            log("Context is tight; retrying with boilerplate stripped from the documents",
                indent=1)
            pack.compress()
            estimate = pack.estimate_tokens(len(system_prompt))
            log.kv("prompt estimate", f"{estimate:,} tokens after compression", indent=1)

        try:
            observations = classify_photos(pack, PHOTO_CLASSES, args, log)
            log("photographic evidence:", indent=1)
            for line in render_observations(observations).splitlines():
                log(line, indent=2)
            parsed = analyse(pack, observations, system_prompt, args, n_ctx, log)
        except ServerError as exc:
            log(f"FAILED {case_dir.name}: {exc}")
            failures.append((case_dir.name, str(exc)))
            continue
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            log(f"FAILED {case_dir.name}: transport error {exc!r}")
            failures.append((case_dir.name, repr(exc)))
            continue
        except (ValueError, json.JSONDecodeError) as exc:
            log(f"FAILED {case_dir.name}: could not parse the verdict -- {exc}")
            failures.append((case_dir.name, f"parse: {exc}"))
            continue

        log.kv("verdict", parsed.get("verdict"))
        log.kv("fraud probability", parsed.get("fraud_probability"))
        log.kv("confidence", parsed.get("confidence"))
        log.kv("primary finding",
               (parsed.get("primary_finding") or {}).get("one_line", "--"))
        log.kv("detectors fired", ", ".join(
            f"{d.get('detector')}" for d in fired_detectors(parsed)) or "none")
        log.kv("action", parsed.get("recommended_action"))

        log.write_file(args.raw_dir / f"{case_dir.name}.json",
                       json.dumps(parsed, indent=2, ensure_ascii=False),
                       what="raw verdict JSON")
        results.append((case_dir.name, parsed))

    log.rule("REPORT")
    report = build_report(results, args, n_ctx, len(system_prompt), started,
                          time.time() - clock, vision is None)
    log.write_file(args.out, report, what="markdown report")

    log("")
    log(f"Cases processed : {len(results)}/{len(case_dirs)}")
    log(f"Failures        : {len(failures)}")
    for name, reason in failures:
        log(f"  {name}: {reason}")
    log(f"Elapsed         : {(time.time() - clock) / 60:.1f} min")
    log(f"Report          : {args.out}")
    log(f"Raw JSON        : {args.raw_dir}")
    log.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
