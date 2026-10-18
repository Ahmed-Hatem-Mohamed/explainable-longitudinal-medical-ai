# NACC Progression Modeling Modules

This package implements the model, training, comparison, and explainability layer for the NACC V1 research prototype. It consumes preprocessed `PatientSequence` objects; it does not request, parse, or store restricted NACC source files. Do not put NACC IDs, participant-level data, or restricted model artifacts in version control.

## Installation

From the repository root, create an isolated environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

A compatible PyTorch installation may require a platform-specific wheel. See the official PyTorch installation selector if the default package installation is not suitable for the machine.

## Input Contract

`PatientSequence` represents one pseudonymous participant and one prediction anchor:

| Field | Shape | Contract |
|---|---|---|
| `subject_key` | scalar | Internal pseudonymous key; never a raw NACCID. |
| `label` | scalar | `1` for MCI progressing to Alzheimer dementia within 730 days after the anchor, `0` for a qualifying negative. |
| `imaging_values` | `[T, M]` | Numeric regional MRI measures after training-only imputation and scaling. The ordered feature dictionary must be identical across every partition. |
| `imaging_missing_mask` | `[T, M]` | Boolean; `True` means the original measurement was missing before imputation. |
| `clinical_values` | `[T, 6]` | Ordered as age, sex encoding, years of education, CDR global, CDR sum of boxes, and MMSE; imputed/transformed using training-only parameters. |
| `clinical_missing_mask` | `[T, 6]` | Boolean; `True` means the original clinical value was missing before imputation. |
| `delta_time` | `[T, 1]` | Elapsed years since the previous visit; first visit is exactly zero and later intervals are non-negative. |

`T` is at most three for this V1. The model accepts a one-visit prefix for inference, although the final anchor cohort should follow the agreed inclusion rule. Inputs must be finite. The data pipeline must fit imputation, feature filtering, and learned scaling on training participants only, then reuse those fitted parameters unchanged for validation and test.

The member responsible for data construction should adapt their output to `PatientSequence` at one boundary. The MRI feature order must be stored alongside the existing MRI feature dictionary and reused unchanged for every split and checkpoint. The clinical encoding and category map (including the sex encoding) must also be recorded. No model or evaluation function reads CSV/Parquet files directly.

## Model Semantics

`MultimodalGRU` implements the specified regional-MRI encoder, clinical encoder, visit-level concatenation, elapsed-time input, one-layer GRU, and binary prediction head. `ModelConfig` supports clinical-only and MRI-only GRU variants for longitudinal comparisons.

The label belongs to the final anchor. Training applies binary cross-entropy only to the last valid logit. Earlier prefix scores show how the model's estimate of that same anchor outcome changes as history is supplied; they are not calibrated 24-month risks at each earlier visit date. A separate per-visit horizon label and follow-up rule would be required to make that latter claim.

Packed sequences prevent padded visits from affecting GRU states. Batches use zero padding and a contiguous-prefix boolean mask. The API accepts only already-imputed values plus their missingness masks.

## Training and Evaluation

```python
from nacc_progression.evaluation import run_model_comparison, write_evaluation_report
from nacc_progression.model import ModelConfig
from nacc_progression.training import TrainingConfig

results = run_model_comparison(
    training_sequences=train_sequences,
    validation_sequences=validation_sequences,
    test_sequences=test_sequences,
    model_config=ModelConfig(
        imaging_features=number_of_mri_features,
        imaging_feature_names=ordered_mri_feature_names,
    ),
    training_config=TrainingConfig(seed=42),
    checkpoint_dir="artifacts/checkpoints",
)
write_evaluation_report(results, "reports/metrics/model_comparison.json")
```

The comparison runs six fixed arms on identical patient partitions:

- Logistic regression on clinical variables from the anchor visit only.
- Logistic regression on MRI measures from the anchor visit only.
- Logistic regression on both modalities from the anchor visit only.
- Clinical-only longitudinal GRU.
- MRI-only longitudinal GRU.
- Multimodal longitudinal GRU.

Logistic inputs include the corresponding missingness indicators. Each GRU uses the same architecture family and training protocol; the single-modality configurations omit the other encoder. Train/validation/test subject keys must be disjoint. The training routine accepts no test data. For each arm, a balanced-accuracy threshold is selected on validation before test predictions are generated. The test partition is used only for final reported metrics and patient-cluster bootstrap intervals.

Reported metrics include AUROC, AUPRC, balanced accuracy, sensitivity, specificity, F1, Brier score, and equal-width-bin ECE. JSON reports contain aggregate results only; they do not include participant keys or per-patient predictions. `bootstrap_confidence_intervals` resamples by participant key, so repeated anchors from one participant remain in the same bootstrap cluster.

## Explainability

- `explain_patient` computes trapezoidal Integrated Gradients for regional MRI values, clinical values, and elapsed time. Its default baseline is zero in normalized value space, while missingness masks are held fixed. Attributions therefore describe value contributions conditional on the observed missingness pattern; they do not attribute the missingness indicators themselves.
- MRI attributions refer to named regional measurements. This V1 does not operate on raw MRI voxels and does not generate voxel-wise saliency maps.
- `leave_one_visit_out` reports output changes after removing each visit and bridging elapsed time. These are model-behavior diagnostics; removing the anchor still evaluates against the original anchor outcome.
- `assess_feature_ablation_fidelity` compares Integrated Gradients feature rankings with the probability changes caused by zeroing one regional measure or clinical variable at a time. Zeroing can create out-of-distribution inputs; agreement is not causal or clinical validation.
- `assess_explanation_stability` compares attribution rankings under small Gaussian perturbations of observed normalized values. Additional models trained with different seeds can be passed to assess retraining stability. Both checks are diagnostics and do not establish clinical validity or causality.

Feature names should be passed to `explain_patient` from the MRI feature dictionary and the clinical schema. Keep that dictionary and the fitted preprocessing artifacts paired with the checkpoint used for inference.
