"""Shared data contracts for model, evaluation, and application integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


CLINICAL_FEATURE_COUNT = 6
MAX_VISITS = 3


@dataclass(frozen=True)
class PatientSequence:
    """One patient anchor and its ordered, preprocessed visit history.

    Values must already be imputed and transformed using training-only
    preprocessing parameters. ``delta_time`` is elapsed time in years since the
    previous visit; its first element is zero. The outcome label belongs to the
    final visit (the prediction anchor), not to every prefix independently.
    """

    subject_key: str
    label: int
    imaging_values: np.ndarray
    imaging_missing_mask: np.ndarray
    clinical_values: np.ndarray
    clinical_missing_mask: np.ndarray
    delta_time: np.ndarray
    imaging_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.subject_key:
            raise ValueError("subject_key must be a non-empty pseudonymous key")
        if self.label not in (0, 1):
            raise ValueError("label must be binary (0 or 1)")

        arrays = {
            "imaging_values": np.asarray(self.imaging_values, dtype=np.float32),
            "imaging_missing_mask": np.asarray(self.imaging_missing_mask, dtype=np.bool_),
            "clinical_values": np.asarray(self.clinical_values, dtype=np.float32),
            "clinical_missing_mask": np.asarray(self.clinical_missing_mask, dtype=np.bool_),
            "delta_time": np.asarray(self.delta_time, dtype=np.float32),
        }
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

        imaging = arrays["imaging_values"]
        imaging_mask = arrays["imaging_missing_mask"]
        clinical = arrays["clinical_values"]
        clinical_mask = arrays["clinical_missing_mask"]
        delta_time = arrays["delta_time"]
        feature_names = tuple(self.imaging_feature_names)
        object.__setattr__(self, "imaging_feature_names", feature_names)

        if imaging.ndim != 2 or imaging.shape[0] == 0:
            raise ValueError("imaging_values must have shape [visits, MRI features]")
        visit_count, feature_count = imaging.shape
        if visit_count > MAX_VISITS:
            raise ValueError(f"V1 supports at most {MAX_VISITS} visits per sequence")
        if feature_count == 0:
            raise ValueError("at least one MRI feature is required")
        if feature_names and len(feature_names) != feature_count:
            raise ValueError("imaging_feature_names must match the MRI feature count")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("imaging_feature_names must be unique")
        if imaging_mask.shape != imaging.shape:
            raise ValueError("imaging_missing_mask must match imaging_values")
        if clinical.shape != (visit_count, CLINICAL_FEATURE_COUNT):
            raise ValueError(
                f"clinical_values must have shape [visits, {CLINICAL_FEATURE_COUNT}]"
            )
        if clinical_mask.shape != clinical.shape:
            raise ValueError("clinical_missing_mask must match clinical_values")
        if delta_time.shape != (visit_count, 1):
            raise ValueError("delta_time must have shape [visits, 1]")
        for name, value in (
            ("imaging_values", imaging),
            ("clinical_values", clinical),
            ("delta_time", delta_time),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite, preprocessed values")
        if delta_time[0, 0] != 0 or np.any(delta_time < 0):
            raise ValueError("delta_time must be non-negative and start at zero")

    @property
    def visit_count(self) -> int:
        return int(self.imaging_values.shape[0])

    @property
    def imaging_feature_count(self) -> int:
        return int(self.imaging_values.shape[1])


@dataclass
class PatientBatch:
    """Padded tensor batch. Valid visits occupy a contiguous prefix per patient."""

    subject_keys: list[str]
    imaging_values: Tensor
    imaging_missing_mask: Tensor
    clinical_values: Tensor
    clinical_missing_mask: Tensor
    delta_time: Tensor
    sequence_mask: Tensor
    labels: Tensor

    @property
    def sequence_lengths(self) -> Tensor:
        return self.sequence_mask.sum(dim=1).to(dtype=torch.long)

    def to(self, device: torch.device | str) -> PatientBatch:
        return PatientBatch(
            subject_keys=self.subject_keys,
            imaging_values=self.imaging_values.to(device),
            imaging_missing_mask=self.imaging_missing_mask.to(device),
            clinical_values=self.clinical_values.to(device),
            clinical_missing_mask=self.clinical_missing_mask.to(device),
            delta_time=self.delta_time.to(device),
            sequence_mask=self.sequence_mask.to(device),
            labels=self.labels.to(device),
        )


@dataclass(frozen=True)
class PredictionResult:
    """Prediction plus prefix scores with explicit anchor-label semantics."""

    subject_key: str
    probability: float
    threshold: float
    predicted_label: int
    prefix_probabilities: tuple[float, ...]
    prefix_semantics: str = (
        "Scores for the anchor outcome as progressively longer history is supplied; "
        "not separately calibrated future risks at each prefix date."
    )


def collate_patient_sequences(sequences: Sequence[PatientSequence]) -> PatientBatch:
    """Pad a list of sequences with zeros and build a valid-prefix mask."""

    if not sequences:
        raise ValueError("cannot collate an empty sequence list")
    feature_counts = {sequence.imaging_feature_count for sequence in sequences}
    if len(feature_counts) != 1:
        raise ValueError("all sequences in a batch must have the same MRI feature count")

    batch_size = len(sequences)
    max_visits = max(sequence.visit_count for sequence in sequences)
    imaging_count = sequences[0].imaging_feature_count
    imaging = np.zeros((batch_size, max_visits, imaging_count), dtype=np.float32)
    imaging_mask = np.zeros_like(imaging, dtype=np.bool_)
    clinical = np.zeros((batch_size, max_visits, CLINICAL_FEATURE_COUNT), dtype=np.float32)
    clinical_mask = np.zeros_like(clinical, dtype=np.bool_)
    delta_time = np.zeros((batch_size, max_visits, 1), dtype=np.float32)
    sequence_mask = np.zeros((batch_size, max_visits), dtype=np.bool_)

    for index, sequence in enumerate(sequences):
        length = sequence.visit_count
        imaging[index, :length] = sequence.imaging_values
        imaging_mask[index, :length] = sequence.imaging_missing_mask
        clinical[index, :length] = sequence.clinical_values
        clinical_mask[index, :length] = sequence.clinical_missing_mask
        delta_time[index, :length] = sequence.delta_time
        sequence_mask[index, :length] = True

    return PatientBatch(
        subject_keys=[sequence.subject_key for sequence in sequences],
        imaging_values=torch.from_numpy(imaging),
        imaging_missing_mask=torch.from_numpy(imaging_mask),
        clinical_values=torch.from_numpy(clinical),
        clinical_missing_mask=torch.from_numpy(clinical_mask),
        delta_time=torch.from_numpy(delta_time),
        sequence_mask=torch.from_numpy(sequence_mask),
        labels=torch.tensor([sequence.label for sequence in sequences], dtype=torch.float32),
    )