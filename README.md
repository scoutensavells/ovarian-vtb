# Agentic AI-orchestrated virtual tumor boards for ovarian cancer

This repository contains the Multi-Agent Structured Deliberation Protocol used to coordinate pathology, radiology, medical oncology, and molecular biology assessments for ovarian cancer cases. The implementation covers structured case representation, evidence weighting, three-round deliberation, conflict resolution, split conformal prediction, calibration, paired statistical comparison, reader-study analysis, and silent-mode deployment auditing.

The software is intended for research evaluation with de-identified records. It does not provide medical advice and must not be used to direct patient care.

## Protocol

Each case contains five sections:

- Demographics
- Pathology summary
- Imaging summary
- Genomic profile
- Clinical history

Each specialist supplies a structured assessment containing a primary label, confidence, candidate distribution, cited evidence, and unresolved uncertainties. The protocol then applies three phases:

1. Independent specialist assessment
2. Evidence-weighted structured debate
3. Conflict resolution and ranked consensus

The permitted epistemic acts are `propose`, `challenge`, `support`, and `synthesize`. Deliberation ends after at most three rounds. A normalized confidence span below `0.15` indicates convergence. Larger disagreements are classified as incomplete evidence, interpretive disagreement, or clinical equipoise.

For candidate label `y`, split conformal prediction uses the nonconformity score `1 - p(y | x)`. The default significance level is `0.10`. The public benchmark uses 410 development cases for calibration and 175 temporally held-out cases for evaluation. The prospective layer uses the 3,847-case retrospective arm for calibration and the 1,462-case silent-mode cohort for evaluation.

## Installation

Create an isolated environment with Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install .
```

Conda users can create the pinned environment:

```bash
conda env create -f environment.yml
conda activate msdp-board
```

The container image uses Python 3.11 on Debian Bookworm:

```bash
docker build -t msdp-board .
```

No CUDA runtime is required. The primary method calls a hosted language-model service in the study design, and the released analytical pipeline consumes already structured specialist assessments. The manuscript does not report GPU count, batch size, learning rate, optimizer, epoch count, or a local training schedule.

## Data

Verified public dataset addresses are listed in `datasets.txt`.

The public benchmark combines:

- TCGA-OV genomic and clinical records from the Genomic Data Commons
- Linked TCGA-OV imaging from The Cancer Imaging Archive
- Supplemental non-serous genomic and clinical records from cBioPortal

The combined public cohort contains 585 cases. Temporal ordering produces a 410-case development partition and a 175-case holdout partition. The five reported histotypes are high-grade serous, endometrioid, clear cell, mucinous, and low-grade serous ovarian carcinoma.

Dataset licenses and access restrictions are controlled by the source portals and individual studies. Review those terms before downloading. Clinical cohorts from the four participating centers are not public and are deliberately excluded from download tooling.

Input records must be de-identified JSON Lines documents. The schema rejects unexpected fields, malformed probability distributions, invalid dates, non-contiguous ranks, missing specialist roles, and empty conformal sets.

## Evaluation

Build an evaluation report from case-level records:

```bash
msdp-evaluate input.jsonl report.json
```

Generate the prespecified prospective and deployment-safety criteria:

```bash
msdp-audit audit.json
```

The analytical layer includes:

- Micro-averaged diagnostic accuracy
- Guideline-concordant decision rate
- Macro one-vs-rest histotype AUC
- Fifteen-bin expected calibration error
- Brier score and reliability-resolution decomposition
- Conformal coverage and prediction-set size
- Abstention precision, recall, and silent-failure rate
- One thousand-iteration case-level bootstrap intervals
- Exact paired McNemar comparisons with Bonferroni correction
- Paired DeLong AUC comparison
- Site, region, age-tercile, and histotype subgroup analysis
- Quarterly stability and modality-removal perturbation analysis
- Three-arm reader-study summaries
- Workflow availability, consultation, agreement, and preparation time

Stochastic methods are aggregated over five independent runs. Traditional rule-based decision support is treated as deterministic. The familywise comparison level is `0.05` across five prespecified method comparisons, producing an adjusted threshold of `0.01`.

## Expected study values

The public temporal holdout contains 175 cases. The reported primary values for MSDP are diagnostic accuracy `91.2%`, guideline-concordant decision rate `89.7%`, histotype AUC `0.904`, ECE `0.042`, and conformal coverage `91.4%` at `alpha=0.10`.

The prospective silent-mode cohort contains 1,462 consecutively enrolled cases. The reported primary values are diagnostic accuracy `90.1%`, tumor-board concordance `89.4%`, guideline-concordant decision rate `88.6%`, and conformal coverage `90.8%`.

The reader study uses a stratified 420-case subset and six board-certified gynecologic oncologists. Reported diagnostic accuracy is `86.7%` for clinicians alone, `90.1%` for the model alone, and `93.8%` for the combined team.

## Safety boundaries

The silent-mode audit evaluates failure taxonomy, potential harm, subgroup performance, temporal stability, perturbation response, uncertainty coverage, abstention, and high-confidence errors. The prespecified thresholds are:

- Significant potential harm below `1%`
- Minimum subgroup accuracy at least `80%`
- Maximum quarterly accuracy change below `3` percentage points
- Accuracy loss under perturbation below `5` percentage points
- Conformal coverage at least `90%`
- Silent-failure rate below `3%`

All outputs preserve case keys only. Do not place names, email addresses, telephone numbers, medical-record numbers, local filesystem paths, or free-text identifiers in input records. The privacy scanner rejects common direct identifiers before report generation.

## Package layout

The `domain` package defines cases and specialist assessments. `protocol` contains weighting, debate state, conflict classification, consensus, and trace transitions. `uncertainty`, `metrics`, and `statistics` implement the numerical methods. `evaluation` contains cohort, subgroup, safety, reader, and workflow analysis. `audit` encodes prespecified criteria. `records` provides strict exchange schemas. `reporting` creates atomic JSON and CSV outputs. `governance` separates verified public sources from restricted clinical cohorts.

## Limitations

The public datasets do not include the prospective or retrospective institutional records. Clinical guideline text is not redistributed. Language-model calls and hidden reasoning are outside this repository. Results on public cases do not establish suitability for clinical use, and exchangeability remains necessary for the stated conformal coverage guarantee.
