# Legal Judgment Prediction Robustness via Transferred Section Structure

## What This Repository Is

This repository is the full research and paper-support codebase for a legal judgment prediction robustness project built on the IL-TUR data setting. The project starts from a practical problem:

legal judgments are long, internally structured documents, but many prediction systems treat them as plain unstructured text. That makes it hard to answer a basic but important question:

**which parts of a judgment actually matter for prediction, and what happens when those parts are removed or isolated?**

The codebase answers that question through a section-aware pipeline built around:

- rhetorical-role supervision from `RR`
- transfer of broad section labels onto `CJPE`
- pseudo-section reconstruction for CJPE cases
- judgment-prediction baselines over pseudo-sectioned CJPE
- section-aware perturbation evaluation
- section-importance quantification
- paper-facing results, figures, and claim-to-evidence traceability

The repository is no longer in infrastructure-building mode. It has already reached the final paper-support stage and now contains:

- the canonical frozen evidence package
- the canonical paper-drafting package
- the canonical section-importance package
- the final submission-style writing bundle

This means the project is already ready for pilot manuscript writing.

## The Main Research Question

The paper asks:

**Can we estimate how sensitive legal judgment prediction is to different parts of a judgment when gold section annotation is unavailable?**

The approach taken here is:

1. use RR as the supervision source for broad legal structure
2. learn a broad-section tagger on RR
3. transfer that structure onto CJPE
4. reconstruct pseudo-sections for CJPE
5. run prediction and section-aware perturbation experiments on pseudo-sectioned CJPE
6. quantify which sections seem most important for prediction

In short:

this is a robustness paper about **document structure**, **model dependence on different judgment sections**, and **whether section-aware perturbations reveal meaningful vulnerabilities**.

## The Core Scientific Idea

Gold section annotation is not available for CJPE in the form needed for this project. Instead of abandoning section-aware robustness analysis, the pipeline uses a structured transfer approach:

- RR provides section/rhetorical supervision
- RR labels are mapped into broad sections:
  - `facts`
  - `precedents`
  - `reasoning`
  - `conclusion`
  - `other`
- a broad-section tagger is trained on RR
- that tagger predicts section labels for CJPE sentences
- sentence predictions are reconstructed into pseudo-sections at the case level

These pseudo-sections are **not gold**. They are transferred/predicted structure. That caveat must remain attached to every scientific claim in this project.

But they are good enough for:

- pilot section-aware perturbation work
- section masking and ablation
- baseline robustness comparisons
- section-importance ranking

They are **not** good enough to claim gold structural analysis of CJPE.

## Final Frozen Scientific Story

The final paper-facing story in this repository is intentionally narrow and stable.

### Main prediction model

The primary paper-facing model is:

- `averaged_passive_aggressive::pseudo_all_sections`

This is referred to throughout the paper package as the main APA model.

### Main perturbation probes

The paper is centered on two perturbations:

1. `keep_reasoning_only`
2. `drop_precedents`

These were chosen because they are the strongest and most interpretable probes after coverage and robustness analysis.

### Main section-importance ranking

Under the frozen APA-centered section-importance package, the ranking is:

1. `precedents`
2. `facts`
3. `reasoning`
4. `other`
5. `conclusion`

Important caveat:

- `conclusion` is low-confidence because conclusion coverage is sparse in the transferred CJPE pseudo-sections.

## Final Findings

This is the shortest honest summary of what the project found.

### 1. APA is the strongest main baseline

On the final frozen `pseudo_all_sections` test setting:

- APA accuracy: `0.574819`
- APA macro F1: `0.565853`

Comparator results:

- NB accuracy: `0.561635`, macro F1: `0.547074`
- logistic accuracy: `0.513514`, macro F1: `0.387077`
- contextual approximation accuracy: `0.510877`, macro F1: `0.377109`

What this means:

- APA is the best single central model for reporting the paper’s main result
- NB remains an important supporting comparator because it often retains performance well under perturbation
- logistic remains an important supporting comparator because it is often the most stable by flip rate
- the current contextual approximation does not beat the strongest simple baselines, so it is contextual background rather than the main result

### 2. Reasoning-only causes the largest main degradation

For APA:

- unperturbed macro F1: `0.565853`
- `keep_reasoning_only` macro F1: `0.515251`
- delta macro F1: `-0.050602`
- unperturbed accuracy: `0.574819`
- `keep_reasoning_only` accuracy: `0.546473`
- delta accuracy: `-0.028346`
- flip rate: `0.334212`
- coverage: `1.0`

What this means:

- reasoning alone still preserves a substantial amount of signal
- but removing facts and precedents still hurts materially
- reasoning sections are important, but the full predictive signal is not reducible to reasoning text alone
- this is the strongest single probe in the paper because it is both high-coverage and behaviorally informative

### 3. Dropping precedents causes a smaller but still meaningful degradation

For APA:

- unperturbed macro F1: `0.565853`
- `drop_precedents` macro F1: `0.548445`
- delta macro F1: `-0.017408`
- unperturbed accuracy: `0.574819`
- `drop_precedents` accuracy: `0.564272`
- delta accuracy: `-0.010547`
- flip rate: `0.145023`
- coverage: `0.845089`

What this means:

- precedents are not ignorable supporting context
- removing precedent material changes predictions in a non-trivial fraction of cases
- the degradation is smaller than reasoning-only ablation, but it is still meaningful and well-supported by coverage
- this makes `drop_precedents` the best secondary supporting probe in the paper

### 4. Robustness, retention, and stability are not the same thing

The project explicitly found a tension between:

- strongest absolute performance
- best metric retention under perturbation
- lowest flip rate

In the frozen package:

- APA is strongest in absolute performance
- NB is often strongest by perturbation-retention delta
- logistic is often strongest by flip-rate stability

What this means:

- a model can be stronger overall but still more unstable
- a model can be stable but much weaker
- robustness claims in the paper must be nuanced, not collapsed into a single metric

That is why the paper package includes:

- main performance tables
- flip-rate figures
- stability-vs-correctness analysis
- qualitative examples

### 5. Section importance is not the same as perturbation ranking, but it complements it

The final APA-centered section-importance table reports:

| Section | Rank | Drop Delta F1 | Keep-only Retention | Drop Flip | Coverage | Confidence | Composite |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| precedents | 1 | `-0.021091` | `0.992377` | `0.171607` | `0.845089` | high | `0.931471` |
| facts | 2 | `0.007817` | `0.943093` | `0.261056` | `0.924192` | high | `0.532618` |
| reasoning | 3 | `-0.001669` | `0.942170` | `0.057389` | `0.918919` | high | `0.411869` |
| other | 4 | `-0.001058` | `0.905864` | `0.000923` | `0.714568` | high | `0.342769` |
| conclusion | 5 | `0.000000` | `0.576162` | `0.000000` | `0.026368` | low | `0.203206` |

What this means:

- under the APA-centered scoring design, precedent material appears most important overall
- facts retain strong standalone signal and rank above reasoning in the composite
- reasoning is still clearly important, but in this setup it does not dominate the ranking
- conclusion should not be overinterpreted because coverage is too sparse

### 6. Cross-model alignment on section importance is only partial

The supporting cross-model check found:

- `3/3` models support: precedent-only retention exceeds facts-only retention
- `1/3` models support: reasoning removal hurts more than precedent removal
- `0/3` models support: reasoning-only retention exceeds facts-only retention

What this means:

- the section-importance story is real enough to report
- but it should be described as **APA-centered pilot evidence**, not as a universal law across all models

## What The Findings Mean At A Higher Level

The project supports the following paper-level interpretation:

1. legal judgment prediction is meaningfully structure-sensitive
2. reasoning content alone carries large predictive signal, but not all of it
3. precedent content matters more than a purely decorative or citation-only view would suggest
4. model strength, robustness retention, and stability must be distinguished
5. transferred pseudo-sections are sufficient for pilot structure-aware robustness analysis
6. the resulting claims are promising, but they remain pilot-scale and should be described with care

## What Was Kept In The Repository

The repository has been pruned to keep the files that matter for:

- rerunning the pipeline
- inspecting the frozen evidence
- writing the paper

The kept core is:

- `configs/`
- `scripts/`
- `src/`
- `tests/`
- `pyproject.toml`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `README.md`
- `PIVOT_CLEANUP_REPORT.md`

The kept canonical artifact runs are:

- section transfer:
  - `outputs/reports/section_transfer/20260403T081506Z_section_transfer_section-transfer-mainline-pivot`
- baseline training:
  - `outputs/reports/prediction_baselines/20260405T031148Z_prediction_baselines_contextual-focused-baselines-v1`
- robustness results package:
  - `outputs/reports/robustness/20260405T044043Z_robustness_manuscript-ready-results-package-v1`
- paper drafting freeze:
  - `outputs/reports/paper_drafting_package/20260405T054818Z_paper_drafting_package_paper-drafting-freeze-v1`
- section importance:
  - `outputs/reports/section_importance/20260405T061707Z_section_importance_section-importance-v2`
- final submission-style paper package:
  - `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2`

The following classes of material were removed or minimized:

- superseded intermediate report runs
- test/temp folders
- cache folders
- `__pycache__`

Some local runtime support folders are still present because they can help execution on this machine:

- `.venv`
- `.deps`
- `.uv-python`
- `.uv-cache`

These are local environment aids, not part of the scientific contribution.

## Repository Layout

```text
Final/
  configs/
    base.yaml
  scripts/
    inspect_dataset.py
    run_section_transfer.py
    train_baseline.py
    evaluate_robustness.py
    export_paper_drafting_package.py
    run_section_importance.py
    export_submission_package.py
  src/
    legal_robustness/
      config/
      data/
      prediction/
      perturbations/
      section_transfer/
      robustness/
      utils/
      ...
  tests/
  outputs/
    reports/
      section_transfer/
      prediction_baselines/
      robustness/
      paper_drafting_package/
      section_importance/
      submission_package/
```

## What Each Script Does

### `scripts/inspect_dataset.py`

Purpose:

- inspects the IL-TUR dataset root
- validates directory structure
- emits inspection metadata

Use this first if you are unsure whether the dataset root is wired correctly.

### `scripts/run_section_transfer.py`

Purpose:

- loads RR and CJPE
- normalizes them
- exports RR sentence supervision
- segments CJPE sentences
- trains the RR broad-section tagger
- predicts broad sections on CJPE
- reconstructs pseudo-sections
- writes section-transfer readiness artifacts

This is the core RR→CJPE structure-transfer pipeline.

### `scripts/train_baseline.py`

Purpose:

- trains the prediction baselines on pseudo-sectioned CJPE
- exports metrics, predictions, and comparison artifacts

The final canonical baseline run contains:

- logistic
- NB
- APA
- contextual approximation

### `scripts/evaluate_robustness.py`

Purpose:

- evaluates the trained models on section-aware perturbation sets
- exports robustness metrics
- exports the manuscript-ready results package

This is where the main perturbation story is produced.

### `scripts/export_paper_drafting_package.py`

Purpose:

- freezes the canonical evidence
- validates internal consistency
- exports drafting-support bundles for the paper

This package is the bridge from results to actual writing.

### `scripts/run_section_importance.py`

Purpose:

- runs the APA-centered section-importance matrix
- computes composite importance scores
- ranks sections
- exports narrative summaries and chart-ready data

This is the final major technical analysis stage before manuscript writing.

### `scripts/export_submission_package.py`

Purpose:

- renders the final figures
- exports polished writing-support bundles
- builds claim-to-evidence traceability
- exports the final submission-style package manifest
- validates the package end to end

This is the final paper-production support stage.

## Environment Setup

The project is configured for Python `>=3.11`.

### Recommended setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Because `requirements.txt` contains:

```text
-e .[dev]
```

this installs the project in editable mode plus development dependencies.

### If `python` is not on PATH

Use the venv interpreter explicitly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Test command

```powershell
pytest tests -q
```

## Dataset Assumption

The pipeline assumes a local IL-TUR root such as:

```text
C:\Users\ashis\IL-TUR
```

You can pass the dataset root through:

- CLI flags
- config
- environment variables

Useful environment variables:

```text
LEGAL_ROBUSTNESS_DATASET_ROOT=C:\Users\ashis\IL-TUR
LEGAL_ROBUSTNESS_LOG_LEVEL=INFO
```

## Step-By-Step Guide To Run Everything From Scratch

This is the full canonical rerun path.

### Step 1. Create and activate the environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2. Optionally verify the repo is healthy

```powershell
pytest tests -q
```

### Step 3. Optionally inspect the dataset root

```powershell
python scripts/inspect_dataset.py --dataset-root "C:\Users\ashis\IL-TUR"
```

Expected output family:

- `outputs/reports/dataset_inspection/`

### Step 4. Run the RR→CJPE section-transfer pipeline

```powershell
python scripts/run_section_transfer.py --dataset-root "C:\Users\ashis\IL-TUR" --run-name "section-transfer-mainline-pivot"
```

Canonical frozen output:

- `outputs/reports/section_transfer/20260403T081506Z_section_transfer_section-transfer-mainline-pivot`

Key artifacts:

- `rr_sentence_supervision.parquet`
- `cjpe_sentences.parquet`
- `rr_section_tagger_model.pkl`
- `rr_section_tagger_metrics.md`
- `cjpe_predicted_sections.parquet`
- `cjpe_reconstructed_sections.parquet`
- `section_transfer_readiness_summary.md`

### Step 5. Train the prediction baselines

```powershell
python scripts/train_baseline.py --section-transfer-dir "C:\Users\ashis\OneDrive\Desktop\Final\outputs\reports\section_transfer\20260403T081506Z_section_transfer_section-transfer-mainline-pivot" --run-name "contextual-focused-baselines-v1"
```

Canonical frozen output:

- `outputs/reports/prediction_baselines/20260405T031148Z_prediction_baselines_contextual-focused-baselines-v1`

Key artifacts:

- model artifacts
- dev/test predictions
- unperturbed comparison tables
- perturbation set exports

### Step 6. Run the focused robustness evaluation and results package

```powershell
python scripts/evaluate_robustness.py --baseline-run-dir "C:\Users\ashis\OneDrive\Desktop\Final\outputs\reports\prediction_baselines\20260405T031148Z_prediction_baselines_contextual-focused-baselines-v1" --run-name "manuscript-ready-results-package-v1"
```

Canonical frozen output:

- `outputs/reports/robustness/20260405T044043Z_robustness_manuscript-ready-results-package-v1`

Key artifacts:

- `results_package/table_main_results.md`
- `results_package/table_model_comparison.md`
- `results_package/qualitative_examples_primary.md`
- `results_package/appendix_keep_reasoning_only_bundle.md`
- `results_package/appendix_drop_precedents_bundle.md`

### Step 7. Export the paper-drafting freeze package

```powershell
python scripts/export_paper_drafting_package.py --baseline-run-dir "C:\Users\ashis\OneDrive\Desktop\Final\outputs\reports\prediction_baselines\20260405T031148Z_prediction_baselines_contextual-focused-baselines-v1" --robustness-run-dir "C:\Users\ashis\OneDrive\Desktop\Final\outputs\reports\robustness\20260405T044043Z_robustness_manuscript-ready-results-package-v1" --run-name "paper-drafting-freeze-v1"
```

Canonical frozen output:

- `outputs/reports/paper_drafting_package/20260405T054818Z_paper_drafting_package_paper-drafting-freeze-v1`

Key artifacts:

- `paper_freeze_manifest.md`
- `paper_consistency_check.md`
- `draft_support_results.md`
- `paper_reproducibility_commands.md`

### Step 8. Run section-importance quantification

```powershell
python scripts/run_section_importance.py --baseline-run-dir "C:\Users\ashis\OneDrive\Desktop\Final\outputs\reports\prediction_baselines\20260405T031148Z_prediction_baselines_contextual-focused-baselines-v1" --robustness-run-dir "C:\Users\ashis\OneDrive\Desktop\Final\outputs\reports\robustness\20260405T044043Z_robustness_manuscript-ready-results-package-v1" --run-name "section-importance-v2"
```

Canonical frozen output:

- `outputs/reports/section_importance/20260405T061707Z_section_importance_section-importance-v2`

Key artifacts:

- `section_importance_scores.md`
- `section_importance_ranking.md`
- `section_importance_cross_model_check.md`
- chart-ready section-importance JSONs

### Step 9. Export the final submission-style paper package

```powershell
python scripts/export_submission_package.py --baseline-run-dir "C:\Users\ashis\OneDrive\Desktop\Final\outputs\reports\prediction_baselines\20260405T031148Z_prediction_baselines_contextual-focused-baselines-v1" --robustness-run-dir "C:\Users\ashis\OneDrive\Desktop\Final\outputs\reports\robustness\20260405T044043Z_robustness_manuscript-ready-results-package-v1" --paper-drafting-package-dir "C:\Users\ashis\OneDrive\Desktop\Final\outputs\reports\paper_drafting_package\20260405T054818Z_paper_drafting_package_paper-drafting-freeze-v1" --section-importance-dir "C:\Users\ashis\OneDrive\Desktop\Final\outputs\reports\section_importance\20260405T061707Z_section_importance_section-importance-v2" --run-name "submission-package-v2"
```

Canonical frozen output:

- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2`

Key artifacts:

- final figures
- polished paper section support files
- claim traceability
- layout guides
- final consistency report
- final handoff summary

## Fastest Way To Start Writing Right Now

If you do **not** want to rerun everything and only want to write the paper now, open these files first:

1. `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/paper_results_support.md`
2. `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/claim_to_evidence_traceability.md`
3. `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/fig_main_robustness.png`
4. `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/fig_section_importance.png`
5. `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/submission_package_manifest.md`
6. `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/paper_handoff_summary.md`

## The Models In The Final Comparison

### 1. `tfidf_logistic_regression`

- weak overall baseline
- valuable as a stability comparator
- often lowest flip rate

### 2. `multinomial_naive_bayes`

- stronger than logistic
- often best on retention under perturbation
- important supporting baseline

### 3. `averaged_passive_aggressive`

- strongest main model
- central reporting model in the paper
- used for section-importance quantification

### 4. `section_contextual_logistic_regression`

- context-aware approximation, not a full modern transformer
- weaker than APA and NB in this project
- included mainly as context, not as the headline model

## The Perturbation Families

The full project explored multiple perturbation primitives, but the final paper is intentionally focused.

### Primary probe: `keep_reasoning_only`

Meaning:

- keep only the `reasoning` section
- remove facts, precedents, and the rest

Interpretation:

- tests how much prediction signal survives in reasoning alone
- helps answer whether reasoning dominates the prediction task

### Secondary probe: `drop_precedents`

Meaning:

- remove the `precedents` section from the pseudo-sectioned document

Interpretation:

- tests sensitivity to loss of precedent content
- probes dependence on legal-supporting context and citation-like material

### Why not use conclusion as a main probe?

Because conclusion coverage is sparse in the transferred pseudo-sections:

- conclusion importance and conclusion-target perturbations are low-confidence
- they remain documented, but they are not central in the final manuscript package

## Section-Importance Method

The section-importance layer is APA-centered and coverage-aware.

For each broad section, the package computes:

1. removal impact
   - how much performance drops when that section is removed
2. solo sufficiency
   - how much performance is retained when only that section is kept
3. flip sensitivity
   - how unstable predictions become when that section is removed

The transparent composite formula is:

```text
importance = 0.45 * normalized_removal_impact
           + 0.35 * normalized_solo_sufficiency
           + 0.20 * normalized_flip_sensitivity
```

Confidence labels are then attached using coverage.

This is important:

- the section-importance score is **interpretable**
- it is **not** a black-box attribution score
- it is **not** a causal proof of legal reasoning salience
- it is **pilot evidence over pseudo-sections**

## Why The Section Ranking Matters

The section ranking answers a question the earlier robustness package alone could not answer cleanly:

**which parts of the judgment appear most important overall, not just under one probe?**

The current answer is:

- precedents appear most important under APA
- facts come next
- reasoning remains important but is not the top-ranked section in the composite
- conclusion evidence is too sparse to support strong claims

This matters because it refines the manuscript story:

- robustness probes showed that reasoning-only and precedent removal are the best focused perturbations
- section-importance scoring shows that precedent content may be even more globally important than the isolated perturbation story alone suggests

## Main Caveats That Must Stay In The Paper

These caveats are not optional. They must remain visible.

### 1. Pseudo-sections are transferred/predicted, not gold

This is the biggest methodological caveat in the project.

### 2. Conclusion-based estimates are weak

Conclusion coverage is too sparse to support strong interpretation.

### 3. The contextual model is not a strong modern encoder

It is a dependency-compatible approximation, not a transformer result.

### 4. The benchmark is focused, not exhaustive

The paper intentionally narrows its main evidence to the strongest probes rather than claiming a full final robustness benchmark.

### 5. The section-importance ranking is APA-centered

Cross-model agreement is partial, not universal.

## What Files Matter Most For The Paper

If you are writing the paper, these are the most useful files.

### Main text

- `outputs/reports/robustness/20260405T044043Z_robustness_manuscript-ready-results-package-v1/results_package/table_main_results.md`
- `outputs/reports/robustness/20260405T044043Z_robustness_manuscript-ready-results-package-v1/results_package/table_model_comparison.md`
- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/fig_main_robustness.png`
- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/fig_model_comparison.png`
- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/fig_section_importance.png`

### Supporting interpretation

- `outputs/reports/robustness/20260405T044043Z_robustness_manuscript-ready-results-package-v1/results_package/table_stability_vs_correctness.md`
- `outputs/reports/section_importance/20260405T061707Z_section_importance_section-importance-v2/section_importance_scores.md`
- `outputs/reports/section_importance/20260405T061707Z_section_importance_section-importance-v2/section_importance_cross_model_check.md`

### Writing support

- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/paper_abstract_support.md`
- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/paper_intro_support.md`
- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/paper_method_support.md`
- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/paper_results_support.md`
- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/paper_limitations_ethics_support.md`
- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/paper_conclusion_support.md`

### Safety against manuscript drift

- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/claim_to_evidence_traceability.md`
- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/final_package_consistency_check.md`
- `outputs/reports/submission_package/20260405T094720Z_submission_package_submission-package-v2/submission_package_manifest.md`

## Recommended Paper Narrative

If you want the shortest accurate framing, use this:

This paper studies section-aware robustness in legal judgment prediction under transferred broad-section structure. We use RR supervision to infer pseudo-sections for CJPE, train and compare lightweight judgment-prediction baselines, and evaluate focused perturbations that isolate or remove specific judgment sections. The strongest main baseline is an averaged passive-aggressive classifier over pseudo-sectioned CJPE. The strongest primary perturbation is reasoning-only ablation, which causes the largest degradation, while precedent removal causes a smaller but still meaningful decline. A final APA-centered section-importance package suggests that precedents rank first overall, followed by facts and reasoning, although these estimates remain pilot-level because the underlying sections are transferred/predicted rather than gold.

## If You Only Want The Final Answer

Here is the project’s final direct answer.

### Which parts of the judgment matter most?

Under the final APA-centered section-importance package:

1. precedents
2. facts
3. reasoning
4. other
5. conclusion

with conclusion explicitly low-confidence.

### What do the perturbation results mean?

- reasoning carries major signal, but not all of it
- precedents matter enough that removing them hurts
- facts also carry meaningful standalone signal
- model quality, robustness retention, and stability are different axes

### Is the project ready for writing?

Yes.

The technical analysis is complete enough for pilot manuscript drafting. The remaining work is mainly:

- writing the paper
- polishing figure style if needed
- optional appendix compression or formatting

## Bottom Line

This repository now supports a complete paper-facing story:

- transfer structure from RR to CJPE
- build pseudo-sectioned CJPE
- evaluate structure-aware perturbations
- compare baseline models
- quantify section importance
- freeze the evidence
- generate final figures and manuscript-support bundles

The current evidence is strong enough for a pilot paper, provided the manuscript keeps the central caveat visible:

**the section analysis is based on transferred/predicted pseudo-sections rather than gold structural annotation.**
