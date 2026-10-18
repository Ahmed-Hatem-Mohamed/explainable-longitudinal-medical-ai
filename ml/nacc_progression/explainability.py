"""Integrated Gradients, visit ablation, and explanation-stability checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from nacc_progression.contracts import (
    CLINICAL_FEATURE_COUNT,
    PatientBatch,
    PatientSequence,
    collate_patient_sequences,
)
from nacc_progression.model import MultimodalGRU, final_valid_logits
from nacc_progression.training import predict_patient


@dataclass(frozen=True)
class ExplanationResult:
    subject_key: str
    probability: float
    mri_attributions: np.ndarray
    clinical_attributions: np.ndarray
    delta_time_attributions: np.ndarray
    mri_feature_importance: np.ndarray
    clinical_feature_importance: np.ndarray
    visit_importance: np.ndarray
    modality_importance: dict[str, float]
    completeness_delta_logit: float
    mri_feature_names: tuple[str, ...]
    clinical_feature_names: tuple[str, ...]
    baseline_description: str = (
        "Zero normalized values (training mean after z-scoring); observed/missing "
        "masks are held fixed."
    )


@dataclass(frozen=True)
class VisitAblationResult:
    removed_visit_index: int
    probability_without_visit: float
    signed_probability_change: float
    absolute_probability_change: float
    remaining_visit_count: int


def _with_inputs(
    batch: PatientBatch,
    imaging_values: Tensor,
    clinical_values: Tensor,
    delta_time: Tensor,
) -> PatientBatch:
    return PatientBatch(
        subject_keys=batch.subject_keys,
        imaging_values=imaging_values,
        imaging_missing_mask=batch.imaging_missing_mask,
        clinical_values=clinical_values,
        clinical_missing_mask=batch.clinical_missing_mask,
        delta_time=delta_time,
        sequence_mask=batch.sequence_mask,
        labels=batch.labels,
    )


def _explain_in_eval_mode(function):
    def wrapped(model: MultimodalGRU, *args, **kwargs):
        was_training = model.training
        model.eval()
        try:
            return function(model, *args, **kwargs)
        finally:
            model.train(was_training)

    return wrapped


@_explain_in_eval_mode
@torch.enable_grad()
def explain_patient(
    model: MultimodalGRU,
    sequence: PatientSequence,
    mri_feature_names: Sequence[str] | None = None,
    clinical_feature_names: Sequence[str] | None = None,
    steps: int = 64,
) -> ExplanationResult:
    """Explain the anchor logit with trapezoidal Integrated Gradients.

    The baseline is zero in normalized value space, while missingness indicators
    are fixed at the patient's observed pattern. Therefore, attributions describe
    value contributions conditional on that missingness pattern, not the
    contribution of missingness indicators themselves.
    """

    if steps < 2:
        raise ValueError("Integrated Gradients requires at least two steps")
    mri_names = tuple(
        mri_feature_names
        or sequence.imaging_feature_names
        or [f"mri_feature_{index}" for index in range(sequence.imaging_feature_count)]
    )
    clinical_names = tuple(clinical_feature_names or [
        f"clinical_feature_{index}" for index in range(CLINICAL_FEATURE_COUNT)
    ])
    if len(mri_names) != sequence.imaging_feature_count:
        raise ValueError("MRI feature name count does not match the sequence")
    if len(clinical_names) != CLINICAL_FEATURE_COUNT:
        raise ValueError("exactly six clinical feature names are required")

    device = next(model.parameters()).device
    batch = collate_patient_sequences([sequence]).to(device)
    input_imaging = batch.imaging_values.detach()
    input_clinical = batch.clinical_values.detach()
    input_time = batch.delta_time.detach()
    baseline_imaging = torch.zeros_like(input_imaging)
    baseline_clinical = torch.zeros_like(input_clinical)
    baseline_time = torch.zeros_like(input_time)
    gradient_sums = [
        torch.zeros_like(input_imaging),
        torch.zeros_like(input_clinical),
        torch.zeros_like(input_time),
    ]

    for index, alpha in enumerate(torch.linspace(0.0, 1.0, steps, device=device)):
        current_imaging = (
            baseline_imaging + alpha * (input_imaging - baseline_imaging)
        ).requires_grad_(True)
        current_clinical = (
            baseline_clinical + alpha * (input_clinical - baseline_clinical)
        ).requires_grad_(True)
        current_time = (baseline_time + alpha * (input_time - baseline_time)).requires_grad_(True)
        current_batch = _with_inputs(
            batch, current_imaging, current_clinical, current_time
        )
        output = final_valid_logits(model(current_batch), current_batch.sequence_mask).sum()
        gradients = torch.autograd.grad(
            output,
            (current_imaging, current_clinical, current_time),
            allow_unused=True,
        )
        weight = 0.5 if index in (0, steps - 1) else 1.0
        for position, gradient in enumerate(gradients):
            if gradient is not None:
                gradient_sums[position] += weight * gradient

    scale = 1.0 / (steps - 1)
    mri_attributions = (input_imaging - baseline_imaging) * gradient_sums[0] * scale
    clinical_attributions = (
        (input_clinical - baseline_clinical) * gradient_sums[1] * scale
    )
    time_attributions = (input_time - baseline_time) * gradient_sums[2] * scale

    with torch.no_grad():
        input_logit = final_valid_logits(model(batch), batch.sequence_mask)[0]
        baseline_batch = _with_inputs(
            batch, baseline_imaging, baseline_clinical, baseline_time
        )
        baseline_logit = final_valid_logits(model(baseline_batch), batch.sequence_mask)[0]
        probability = float(torch.sigmoid(input_logit).item())
    attribution_sum = (
        mri_attributions.sum()
        + clinical_attributions.sum()
        + time_attributions.sum()
    )
    completeness_delta = float(
        (input_logit - baseline_logit - attribution_sum).detach().item()
    )

    mri_values = mri_attributions[0, : sequence.visit_count].detach().cpu().numpy()
    clinical_values = clinical_attributions[0, : sequence.visit_count].detach().cpu().numpy()
    time_values = time_attributions[0, : sequence.visit_count].detach().cpu().numpy()
    mri_importance = np.abs(mri_values).sum(axis=0)
    clinical_importance = np.abs(clinical_values).sum(axis=0)
    visit_importance = (
        np.abs(mri_values).sum(axis=1)
        + np.abs(clinical_values).sum(axis=1)
        + np.abs(time_values).sum(axis=1)
    )
    modality_importance = {
        "imaging": float(np.abs(mri_values).sum()),
        "clinical": float(np.abs(clinical_values).sum()),
        "elapsed_time": float(np.abs(time_values).sum()),
    }
    return ExplanationResult(
        subject_key=sequence.subject_key,
        probability=probability,
        mri_attributions=mri_values,
        clinical_attributions=clinical_values,
        delta_time_attributions=time_values,
        mri_feature_importance=mri_importance,
        clinical_feature_importance=clinical_importance,
        visit_importance=visit_importance,
        modality_importance=modality_importance,
        completeness_delta_logit=completeness_delta,
        mri_feature_names=mri_names,
        clinical_feature_names=clinical_names,
    )


def leave_one_visit_out(
    model: MultimodalGRU,
    sequence: PatientSequence,
    threshold: float = 0.5,
) -> list[VisitAblationResult]:
    """Measure score changes after dropping each visit and bridging elapsed time.

    Results are model-behavior diagnostics. Removing the anchor visit means the
    reported score still targets the original anchor label, not a newly defined
    outcome at the shortened sequence's final date.
    """

    if sequence.visit_count < 2:
        raise ValueError("leave-one-visit-out requires at least two visits")
    full_probability = predict_patient(model, sequence, threshold).probability
    results = []
    for removed_index in range(sequence.visit_count):
        kept_indices = [
            index for index in range(sequence.visit_count) if index != removed_index
        ]
        adjusted_time = np.zeros((len(kept_indices), 1), dtype=np.float32)
        for position in range(1, len(kept_indices)):
            previous = kept_indices[position - 1]
            current = kept_indices[position]
            adjusted_time[position, 0] = sequence.delta_time[previous + 1 : current + 1].sum()
        ablated = PatientSequence(
            subject_key=sequence.subject_key,
            label=sequence.label,
            imaging_values=sequence.imaging_values[kept_indices],
            imaging_missing_mask=sequence.imaging_missing_mask[kept_indices],
            clinical_values=sequence.clinical_values[kept_indices],
            clinical_missing_mask=sequence.clinical_missing_mask[kept_indices],
            delta_time=adjusted_time,
            imaging_feature_names=sequence.imaging_feature_names,
        )
        ablated_probability = predict_patient(model, ablated, threshold).probability
        difference = ablated_probability - full_probability
        results.append(
            VisitAblationResult(
                removed_visit_index=removed_index,
                probability_without_visit=ablated_probability,
                signed_probability_change=difference,
                absolute_probability_change=abs(difference),
                remaining_visit_count=ablated.visit_count,
            )
        )
    return results


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _importance_vector(explanation: ExplanationResult) -> np.ndarray:
    return np.concatenate(
        (
            np.abs(explanation.mri_attributions).ravel(),
            np.abs(explanation.clinical_attributions).ravel(),
            np.abs(explanation.delta_time_attributions).ravel(),
        )
    )


def _rank_similarity(
    reference: np.ndarray,
    comparison: np.ndarray,
    top_k: int,
) -> tuple[float, float]:
    reference_ranks = _rankdata(reference)
    comparison_ranks = _rankdata(comparison)
    if np.std(reference_ranks) == 0 or np.std(comparison_ranks) == 0:
        spearman = float("nan")
    else:
        spearman = float(np.corrcoef(reference_ranks, comparison_ranks)[0, 1])
    count = min(top_k, reference.size)
    reference_top = set(np.argpartition(reference, -count)[-count:].tolist())
    comparison_top = set(np.argpartition(comparison, -count)[-count:].tolist())
    union = reference_top | comparison_top
    overlap = len(reference_top & comparison_top) / len(union) if union else 1.0
    return spearman, float(overlap)


def assess_explanation_stability(
    model: MultimodalGRU,
    sequence: PatientSequence,
    noise_scale: float = 0.05,
    perturbation_replicates: int = 5,
    integrated_gradients_steps: int = 32,
    top_k: int = 10,
    seed: int = 42,
    retrained_models: Sequence[MultimodalGRU] = (),
) -> dict[str, object]:
    """Assess attribution-rank stability under small input noise and model reruns.

    Noise is added only to observed, normalized values; missingness patterns,
    labels, visit order, and elapsed times are held fixed. Pass models trained
    with different seeds as ``retrained_models`` to add a retraining-stability
    comparison. This is a diagnostic, not a clinical-validity test.
    """

    if noise_scale < 0 or perturbation_replicates < 1 or top_k < 1:
        raise ValueError("noise scale must be non-negative; replicate counts and top_k positive")
    baseline = explain_patient(model, sequence, steps=integrated_gradients_steps)
    reference_vector = _importance_vector(baseline)
    rng = np.random.default_rng(seed)
    rank_correlations = []
    top_k_overlaps = []
    probability_changes = []
    for _ in range(perturbation_replicates):
        imaging = sequence.imaging_values.copy()
        clinical = sequence.clinical_values.copy()
        imaging_observed = ~sequence.imaging_missing_mask
        clinical_observed = ~sequence.clinical_missing_mask
        imaging[imaging_observed] += rng.normal(
            0.0, noise_scale, size=int(imaging_observed.sum())
        )
        clinical[clinical_observed] += rng.normal(
            0.0, noise_scale, size=int(clinical_observed.sum())
        )
        perturbed = PatientSequence(
            subject_key=sequence.subject_key,
            label=sequence.label,
            imaging_values=imaging,
            imaging_missing_mask=sequence.imaging_missing_mask,
            clinical_values=clinical,
            clinical_missing_mask=sequence.clinical_missing_mask,
            delta_time=sequence.delta_time,
            imaging_feature_names=sequence.imaging_feature_names,
        )
        explanation = explain_patient(
            model, perturbed, steps=integrated_gradients_steps
        )
        correlation, overlap = _rank_similarity(
            reference_vector, _importance_vector(explanation), top_k
        )
        rank_correlations.append(correlation)
        top_k_overlaps.append(overlap)
        probability_changes.append(abs(explanation.probability - baseline.probability))

    retraining_correlations = []
    retraining_overlaps = []
    for alternate_model in retrained_models:
        alternate = explain_patient(
            alternate_model, sequence, steps=integrated_gradients_steps
        )
        correlation, overlap = _rank_similarity(
            reference_vector, _importance_vector(alternate), top_k
        )
        retraining_correlations.append(correlation)
        retraining_overlaps.append(overlap)

    return {
        "subject_key": sequence.subject_key,
        "noise_scale_standardized_units": noise_scale,
        "perturbation_replicates": perturbation_replicates,
        "integrated_gradients_steps": integrated_gradients_steps,
        "top_k": top_k,
        "mean_perturbation_spearman": float(np.nanmean(rank_correlations)),
        "mean_perturbation_top_k_jaccard": float(np.mean(top_k_overlaps)),
        "mean_absolute_probability_change": float(np.mean(probability_changes)),
        "retraining_models_compared": len(retrained_models),
        "mean_retraining_spearman": (
            float(np.nanmean(retraining_correlations)) if retraining_correlations else None
        ),
        "mean_retraining_top_k_jaccard": (
            float(np.mean(retraining_overlaps)) if retraining_overlaps else None
        ),
        "completeness_delta_logit": baseline.completeness_delta_logit,
    }


def assess_feature_ablation_fidelity(
    model: MultimodalGRU,
    sequence: PatientSequence,
    explanation: ExplanationResult | None = None,
    top_k: int = 10,
) -> dict[str, object]:
    """Compare feature-attribution rankings with per-feature occlusion effects.

    Each MRI feature or clinical variable is replaced by zero across visits,
    which is the normalized training mean under the documented input contract.
    Missingness masks are kept fixed. Large distribution shifts are possible;
    report this as local model-behavior fidelity, not causal or clinical
    validation.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    explanation = explanation or explain_patient(model, sequence)
    original_probability = explanation.probability

    def compare_group(group: str) -> dict[str, object]:
        if group == "imaging":
            attribution = explanation.mri_feature_importance
            feature_count = sequence.imaging_feature_count
        else:
            attribution = explanation.clinical_feature_importance
            feature_count = CLINICAL_FEATURE_COUNT

        probability_changes = np.zeros(feature_count, dtype=np.float64)
        for feature_index in range(feature_count):
            imaging_values = sequence.imaging_values.copy()
            clinical_values = sequence.clinical_values.copy()
            if group == "imaging":
                imaging_values[:, feature_index] = 0.0
            else:
                clinical_values[:, feature_index] = 0.0
            ablated = PatientSequence(
                subject_key=sequence.subject_key,
                label=sequence.label,
                imaging_values=imaging_values,
                imaging_missing_mask=sequence.imaging_missing_mask,
                clinical_values=clinical_values,
                clinical_missing_mask=sequence.clinical_missing_mask,
                delta_time=sequence.delta_time,
                imaging_feature_names=sequence.imaging_feature_names,
            )
            ablated_probability = predict_patient(model, ablated).probability
            probability_changes[feature_index] = abs(
                ablated_probability - original_probability
            )

        spearman, overlap = _rank_similarity(
            np.asarray(attribution, dtype=np.float64),
            probability_changes,
            top_k,
        )
        names = (
            explanation.mri_feature_names
            if group == "imaging"
            else explanation.clinical_feature_names
        )
        return {
            "spearman_attribution_vs_occlusion": spearman,
            "top_k_jaccard": overlap,
            "features": [
                {
                    "feature": names[index],
                    "integrated_gradients_importance": float(attribution[index]),
                    "absolute_probability_change_when_zeroed": float(
                        probability_changes[index]
                    ),
                }
                for index in range(feature_count)
            ],
        }

    return {
        "subject_key": sequence.subject_key,
        "top_k": top_k,
        "mri_regional_features": compare_group("imaging"),
        "clinical_variables": compare_group("clinical"),
        "interpretation": (
            "Agreement is local model-behavior evidence only; zeroing features can "
            "create out-of-distribution inputs and does not establish causality."
        ),
    }
