"""Last-visit logistic-regression baselines for modality comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from nacc_progression.contracts import PatientSequence


FeatureKind = Literal["clinical", "imaging", "multimodal"]


@dataclass
class LogisticBaseline:
    name: str
    feature_kind: FeatureKind
    estimator: LogisticRegression


def last_visit_features(
    sequences: Sequence[PatientSequence],
    feature_kind: FeatureKind,
) -> np.ndarray:
    """Build a fixed-width, non-longitudinal matrix from each anchor visit.

    Preprocessing and scaling must already have been fitted on training data by
    the data pipeline. Missingness indicators are retained as model features.
    """

    if feature_kind not in ("clinical", "imaging", "multimodal"):
        raise ValueError(f"unsupported feature kind: {feature_kind}")
    rows = []
    for sequence in sequences:
        pieces = []
        if feature_kind in ("imaging", "multimodal"):
            pieces.extend(
                (
                    sequence.imaging_values[-1],
                    sequence.imaging_missing_mask[-1].astype(np.float32),
                )
            )
        if feature_kind in ("clinical", "multimodal"):
            pieces.extend(
                (
                    sequence.clinical_values[-1],
                    sequence.clinical_missing_mask[-1].astype(np.float32),
                )
            )
        rows.append(np.concatenate(pieces).astype(np.float64, copy=False))
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack(rows)


def fit_logistic_baseline(
    training_sequences: Sequence[PatientSequence],
    feature_kind: FeatureKind,
    seed: int = 42,
    max_iter: int = 1000,
) -> LogisticBaseline:
    """Fit an anchor-only logistic regression using training patients only."""

    if not training_sequences:
        raise ValueError("training partition must not be empty")
    labels = np.asarray([sequence.label for sequence in training_sequences], dtype=np.int64)
    if np.unique(labels).size != 2:
        raise ValueError("training data must contain both outcome classes")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
    )
    estimator.fit(last_visit_features(training_sequences, feature_kind), labels)
    return LogisticBaseline(
        name=f"logistic_{feature_kind}_last_visit",
        feature_kind=feature_kind,
        estimator=estimator,
    )


def predict_logistic_probabilities(
    model: LogisticBaseline,
    sequences: Sequence[PatientSequence],
) -> np.ndarray:
    """Return positive-class probabilities in sequence order."""

    if not sequences:
        return np.empty(0, dtype=np.float64)
    return model.estimator.predict_proba(
        last_visit_features(sequences, model.feature_kind)
    )[:, 1].astype(np.float64)
