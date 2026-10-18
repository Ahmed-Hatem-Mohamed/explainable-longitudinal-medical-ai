"""Leakage-aware model comparisons, threshold selection, metrics, and CIs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from nacc_progression.baselines import (
    fit_logistic_baseline,
    predict_logistic_probabilities,
)
from nacc_progression.contracts import PatientSequence
from nacc_progression.model import ModelConfig
from nacc_progression.training import (
    TrainingConfig,
    predict_probabilities,
    train_model,
)


@dataclass(frozen=True)
class ModelEvaluation:
    name: str
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    test_confidence_intervals: dict[str, tuple[float, float]]
    validation_calibration_curve: dict[str, list[float]]
    test_calibration_curve: dict[str, list[float]]
    best_epoch: int | None = None
    checkpoint_path: str | None = None


def select_balanced_accuracy_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    """Choose a validation threshold; ties prefer a threshold near 0.5."""

    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    _validate_predictions(y_true, y_probability)
    if np.unique(y_true).size != 2:
        raise ValueError("validation data must contain both outcome classes")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_probability)))
    scores = [
        balanced_accuracy_score(y_true, y_probability >= threshold)
        for threshold in candidates
    ]
    best_score = max(scores)
    best_candidates = [
        float(threshold)
        for threshold, score in zip(candidates, scores)
        if np.isclose(score, best_score)
    ]
    return min(best_candidates, key=lambda threshold: (abs(threshold - 0.5), threshold))


def _validate_predictions(labels: np.ndarray, probabilities: np.ndarray) -> None:
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be one-dimensional and equally sized")
    if labels.size == 0:
        raise ValueError("evaluation data must not be empty")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    bin_count: int = 10,
) -> float:
    """Equal-width expected calibration error; report bin count with results."""

    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    _validate_predictions(y_true, y_probability)
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    ece = 0.0
    for index in range(bin_count):
        lower, upper = edges[index], edges[index + 1]
        in_bin = (y_probability >= lower) & (
            y_probability <= upper if index == bin_count - 1 else y_probability < upper
        )
        if in_bin.any():
            confidence = float(y_probability[in_bin].mean())
            accuracy = float(y_true[in_bin].mean())
            ece += float(in_bin.mean()) * abs(accuracy - confidence)
    return float(ece)


def calculate_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    calibration_bins: int = 10,
) -> dict[str, float]:
    """Calculate discrimination, threshold, and calibration metrics."""

    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    _validate_predictions(y_true, y_probability)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    y_predicted = y_probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    if np.unique(y_true).size == 2:
        auroc = float(roc_auc_score(y_true, y_probability))
        auprc = float(average_precision_score(y_true, y_probability))
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_predicted))
    else:
        auroc = float("nan")
        auprc = float("nan")
        balanced_accuracy = float("nan")
    return {
        "n": float(y_true.size),
        "prevalence": float(y_true.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "ece": expected_calibration_error(y_true, y_probability, calibration_bins),
        "calibration_bins": float(calibration_bins),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
    }


def calibration_curve_data(
    labels: Sequence[int],
    probabilities: Sequence[float],
    bin_count: int = 10,
) -> dict[str, list[float]]:
    """Return equal-width reliability-diagram points and bin counts."""

    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    _validate_predictions(y_true, y_probability)
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    counts: list[float] = []
    mean_predictions: list[float] = []
    observed_rates: list[float] = []
    for index in range(bin_count):
        lower, upper = edges[index], edges[index + 1]
        in_bin = (y_probability >= lower) & (
            y_probability <= upper if index == bin_count - 1 else y_probability < upper
        )
        counts.append(float(in_bin.sum()))
        mean_predictions.append(
            float(y_probability[in_bin].mean()) if in_bin.any() else float("nan")
        )
        observed_rates.append(
            float(y_true[in_bin].mean()) if in_bin.any() else float("nan")
        )
    return {
        "bin_edges": edges.tolist(),
        "counts": counts,
        "mean_predicted_probability": mean_predictions,
        "observed_event_rate": observed_rates,
    }


def bootstrap_confidence_intervals(
    labels: Sequence[int],
    probabilities: Sequence[float],
    subject_keys: Sequence[str],
    threshold: float,
    repetitions: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs from patient-cluster bootstrap resampling."""

    y_true = np.asarray(labels, dtype=np.int64)
    y_probability = np.asarray(probabilities, dtype=np.float64)
    keys = np.asarray(subject_keys, dtype=str)
    _validate_predictions(y_true, y_probability)
    if keys.shape != y_true.shape:
        raise ValueError("subject_keys must match labels")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    unique_keys = np.unique(keys)
    group_indices = {key: np.flatnonzero(keys == key) for key in unique_keys}
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {
        name: []
        for name in ("auroc", "auprc", "balanced_accuracy", "brier_score")
    }
    for _ in range(repetitions):
        sampled_keys = rng.choice(unique_keys, size=unique_keys.size, replace=True)
        indices = np.concatenate([group_indices[key] for key in sampled_keys])
        if np.unique(y_true[indices]).size < 2:
            continue
        report = calculate_metrics(
            y_true[indices], y_probability[indices], threshold
        )
        for name in collected:
            collected[name].append(report[name])

    intervals = {}
    for name, values in collected.items():
        if values:
            low, high = np.percentile(values, [2.5, 97.5])
            intervals[name] = (float(low), float(high))
        else:
            intervals[name] = (float("nan"), float("nan"))
    return intervals


def _validate_partitions(
    training: Sequence[PatientSequence],
    validation: Sequence[PatientSequence],
    test: Sequence[PatientSequence],
) -> None:
    partitions = (training, validation, test)
    if any(not partition for partition in partitions):
        raise ValueError("train, validation, and test partitions must be non-empty")
    key_sets = [{sequence.subject_key for sequence in partition} for partition in partitions]
    if key_sets[0] & key_sets[1] or key_sets[0] & key_sets[2] or key_sets[1] & key_sets[2]:
        raise ValueError("patient-level leakage: partition subject keys overlap")
    feature_counts = {
        sequence.imaging_feature_count
        for partition in partitions
        for sequence in partition
    }
    if len(feature_counts) != 1:
        raise ValueError("all partitions must use the same ordered MRI feature set")
    feature_name_sets = {
        sequence.imaging_feature_names
        for partition in partitions
        for sequence in partition
    }
    if len(feature_name_sets) != 1:
        raise ValueError("all partitions must use the same ordered MRI feature names")


def _evaluate_one(
    name: str,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    threshold: float,
    validation_sequences: Sequence[PatientSequence],
    test_sequences: Sequence[PatientSequence],
    bootstrap_repetitions: int,
) -> ModelEvaluation:
    return ModelEvaluation(
        name=name,
        threshold=threshold,
        validation_metrics=calculate_metrics(
            validation_labels, validation_probabilities, threshold
        ),
        test_metrics=calculate_metrics(test_labels, test_probabilities, threshold),
        validation_calibration_curve=calibration_curve_data(
            validation_labels, validation_probabilities
        ),
        test_calibration_curve=calibration_curve_data(test_labels, test_probabilities),
        test_confidence_intervals=bootstrap_confidence_intervals(
            test_labels,
            test_probabilities,
            [sequence.subject_key for sequence in test_sequences],
            threshold,
            repetitions=bootstrap_repetitions,
        ),
    )


def run_model_comparison(
    training_sequences: Sequence[PatientSequence],
    validation_sequences: Sequence[PatientSequence],
    test_sequences: Sequence[PatientSequence],
    model_config: ModelConfig,
    training_config: TrainingConfig = TrainingConfig(),
    checkpoint_dir: str | Path | None = None,
    bootstrap_repetitions: int = 1000,
) -> dict[str, ModelEvaluation]:
    """Run common-split logistic and longitudinal modality ablations.

    Six configurations are compared: clinical-only, MRI-only, and multimodal
    logistic models on the anchor visit, plus the corresponding three GRU models
    using the available visit history. Validation selects each threshold; the
    test partition is used only for final metrics and confidence intervals.
    """

    _validate_partitions(training_sequences, validation_sequences, test_sequences)
    if model_config.imaging_features not in {
        sequence.imaging_feature_count
        for partition in (training_sequences, validation_sequences, test_sequences)
        for sequence in partition
    }:
        raise ValueError("model MRI feature count does not match input sequences")
    if model_config.imaging_feature_names and any(
        sequence.imaging_feature_names != model_config.imaging_feature_names
        for partition in (training_sequences, validation_sequences, test_sequences)
        for sequence in partition
    ):
        raise ValueError("model MRI feature order does not match the data partitions")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")

    y_validation = np.asarray([sequence.label for sequence in validation_sequences])
    y_test = np.asarray([sequence.label for sequence in test_sequences])
    if np.unique(y_validation).size != 2:
        raise ValueError("validation partition must contain both outcome classes")

    results: dict[str, ModelEvaluation] = {}
    for feature_kind in ("clinical", "imaging", "multimodal"):
        baseline = fit_logistic_baseline(
            training_sequences,
            feature_kind=feature_kind,
            seed=training_config.seed,
        )
        validation_probabilities = predict_logistic_probabilities(
            baseline, validation_sequences
        )
        threshold = select_balanced_accuracy_threshold(
            y_validation, validation_probabilities
        )
        test_probabilities = predict_logistic_probabilities(baseline, test_sequences)
        name = baseline.name
        results[name] = _evaluate_one(
            name,
            y_validation,
            y_test,
            validation_probabilities,
            test_probabilities,
            threshold,
            validation_sequences,
            test_sequences,
            bootstrap_repetitions,
        )

    output_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for name, use_imaging, use_clinical in (
        ("gru_clinical_longitudinal", False, True),
        ("gru_imaging_longitudinal", True, False),
        ("gru_multimodal_longitudinal", True, True),
    ):
        current_config = replace(
            model_config,
            use_imaging=use_imaging,
            use_clinical=use_clinical,
        )
        checkpoint = output_dir / f"{name}.pt" if output_dir is not None else None
        trained = train_model(
            training_sequences,
            validation_sequences,
            current_config,
            training_config,
            checkpoint_path=checkpoint,
        )
        validation_probabilities, _ = predict_probabilities(
            trained.model, validation_sequences, training_config.batch_size
        )
        threshold = select_balanced_accuracy_threshold(
            y_validation, validation_probabilities
        )
        test_probabilities, _ = predict_probabilities(
            trained.model, test_sequences, training_config.batch_size
        )
        base_result = _evaluate_one(
            name,
            y_validation,
            y_test,
            validation_probabilities,
            test_probabilities,
            threshold,
            validation_sequences,
            test_sequences,
            bootstrap_repetitions,
        )
        results[name] = replace(
            base_result,
            best_epoch=trained.best_epoch,
            checkpoint_path=trained.checkpoint_path,
        )
    return results


def write_evaluation_report(
    results: dict[str, ModelEvaluation],
    output_path: str | Path,
) -> None:
    """Write aggregate-only JSON results; no participant identifiers are stored."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(result) for name, result in results.items()}
    target.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
