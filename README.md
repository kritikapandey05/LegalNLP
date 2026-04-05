# LegalNLP Project

## What This Folder Is

This folder is a slimmed-down, sendable version of the legal judgment prediction robustness repository.

It keeps:

- the runnable source code
- configs
- scripts
- tests
- packaging metadata
- an empty `outputs/reports/` skeleton where new runs can be written

It does **not** keep:

- the IL-TUR dataset
- large generated report runs
- frozen output artifacts
- virtual environments
- local dependency caches
- temporary folders

## Included Structure

```text
LegalNLP/
  README.md
  FULL_PROJECT_README.md
  pyproject.toml
  requirements.txt
  .env.example
  .gitignore
  configs/
  scripts/
  src/
  tests/
  outputs/
    reports/
```

## What you will need Separately

1. a Python environment
2. the IL-TUR dataset on their own machine
3. enough disk space for generated outputs

## Setup

From inside this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

optionally verify the code:

```powershell
pytest tests -q
```

## Required Dataset

The project expects an IL-TUR dataset root, for example:

```text
C:\Users\<user>\IL-TUR
```

You can provide the dataset root through CLI flags or environment variables.

Useful environment variables:

```text
LEGAL_ROBUSTNESS_DATASET_ROOT=C:\Users\<user>\IL-TUR
LEGAL_ROBUSTNESS_LOG_LEVEL=INFO
```

## Step-By-Step Run Order

### 1. Optional dataset inspection

```powershell
python scripts/inspect_dataset.py --dataset-root "C:\Users\<user>\IL-TUR"
```

### 2. Run RR to CJPE section transfer

```powershell
python scripts/run_section_transfer.py --dataset-root "C:\Users\<user>\IL-TUR" --run-name "section-transfer-mainline-pivot"
```

### 3. Train baselines

Replace `<section_transfer_run_dir>` with the directory generated in step 2.

```powershell
python scripts/train_baseline.py --section-transfer-dir "outputs\reports\section_transfer\<section_transfer_run_dir>" --run-name "contextual-focused-baselines-v1"
```

### 4. Run robustness evaluation and results packaging

Replace `<baseline_run_dir>` with the directory generated in step 3.

```powershell
python scripts/evaluate_robustness.py --baseline-run-dir "outputs\reports\prediction_baselines\<baseline_run_dir>" --run-name "manuscript-ready-results-package-v1"
```

### 5. Export the paper drafting package

```powershell
python scripts/export_paper_drafting_package.py --baseline-run-dir "outputs\reports\prediction_baselines\<baseline_run_dir>" --robustness-run-dir "outputs\reports\robustness\<robustness_run_dir>" --run-name "paper-drafting-freeze-v1"
```

### 6. Run section-importance quantification

```powershell
python scripts/run_section_importance.py --baseline-run-dir "outputs\reports\prediction_baselines\<baseline_run_dir>" --robustness-run-dir "outputs\reports\robustness\<robustness_run_dir>" --run-name "section-importance-v2"
```

### 7. Export the final submission-style package

```powershell
python scripts/export_submission_package.py --baseline-run-dir "outputs\reports\prediction_baselines\<baseline_run_dir>" --robustness-run-dir "outputs\reports\robustness\<robustness_run_dir>" --paper-drafting-package-dir "outputs\reports\paper_drafting_package\<paper_drafting_run_dir>" --section-importance-dir "outputs\reports\section_importance\<section_importance_run_dir>" --run-name "submission-package-v2"
```

## Output Location

All generated runs are written under:

```text
outputs/reports/
```

This package includes that folder only as an empty scaffold. The actual artifacts will appear there after the recipient runs the pipeline.

## Important Scientific Caveat

The project’s section-based results are built on **transferred/predicted pseudo-sections**, not gold section annotations. Anyone reusing the code or writing about the results should keep that caveat visible.
