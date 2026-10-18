"""Reproducible anchor-outcome training and inference utilities."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from nacc_progression.contracts import (
    PatientSequence,
    PredictionResult,
    collate_patient_sequences,
)
from nacc_progression.model import ModelConfig, MultimodalGRU, final_valid_logits


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience, and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")


@dataclass
class TrainingResult:
    model: MultimodalGRU
    best_epoch: int
    best_validation_loss: float
    training_losses: list[float]
    validation_losses: list[float]
    positive_class_weight: float
    checkpoint_path: str | None


def _device_for(configured_device: str) -> torch.device:
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _validate_training_data(
    training_sequences: Sequence[PatientSequence],
    validation_sequences: Sequence[PatientSequence],
) -> None:
    if not training_sequences or not validation_sequences:
        raise ValueError("training and validation partitions must both be non-empty")
    training_keys = {sequence.subject_key for sequence in training_sequences}
    validation_keys = {sequence.subject_key for sequence in validation_sequences}
    overlap = training_keys & validation_keys
    if overlap:
        raise ValueError("patient-level leakage: train and validation keys overlap")
    if len({sequence.imaging_feature_count for sequence in (*training_sequences, *validation_sequences)}) != 1:
        raise ValueError("all partitions must use the same ordered MRI feature set")
    feature_name_sets = {
        sequence.imaging_feature_names
        for sequence in (*training_sequences, *validation_sequences)
    }
    if len(feature_name_sets) != 1:
        raise ValueError("all sequences must provide the same ordered MRI feature names")
    labels = {sequence.label for sequence in training_sequences}
    if labels != {0, 1}:
        raise ValueError("training data must contain both outcome classes")


def _make_loader(
    sequences: Sequence[PatientSequence],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        list(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_patient_sequences,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def _last_step_loss(
    model: MultimodalGRU,
    batch,
    criterion: nn.Module,
) -> Tensor:
    logits = model(batch)
    anchor_logits = final_valid_logits(logits, batch.sequence_mask)
    return criterion(anchor_logits, batch.labels)


def _evaluate_loss(
    model: MultimodalGRU,
    sequences: Sequence[PatientSequence],
    config: TrainingConfig,
    device: torch.device,
) -> float:
    loader = _make_loader(sequences, config.batch_size, shuffle=False, seed=config.seed)
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            loss = _last_step_loss(model, batch, nn.BCEWithLogitsLoss())
            losses.append(float(loss.item()) * batch.labels.shape[0])
    return float(sum(losses) / len(sequences))


def train_model(
    training_sequences: Sequence[PatientSequence],
    validation_sequences: Sequence[PatientSequence],
    model_config: ModelConfig,
    training_config: TrainingConfig = TrainingConfig(),
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Train with anchor labels, train-only positive weight, and validation early stopping.

    The validation partition is used for checkpoint selection. No test data are
    accepted by this function, preventing accidental test-set model selection.
    """

    _validate_training_data(training_sequences, validation_sequences)
    if any(sequence.imaging_feature_count != model_config.imaging_features for sequence in training_sequences):
        raise ValueError("model MRI feature count does not match input sequences")
    if model_config.imaging_feature_names and any(
        sequence.imaging_feature_names != model_config.imaging_feature_names
        for sequence in (*training_sequences, *validation_sequences)
    ):
        raise ValueError("model MRI feature order does not match input sequences")

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = _device_for(training_config.device)
    model = MultimodalGRU(model_config).to(device)
    labels = np.asarray([sequence.label for sequence in training_sequences], dtype=np.int64)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("training data must contain positive and negative examples")
    positive_weight = negative_count / positive_count
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    train_loader = _make_loader(
        training_sequences,
        training_config.batch_size,
        shuffle=True,
        seed=training_config.seed,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    training_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _last_step_loss(model, batch, criterion)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch.labels.shape[0]

        training_loss = total_loss / len(training_sequences)
        validation_loss = _evaluate_loss(model, validation_sequences, training_config, device)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a model checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    checkpoint_name = None
    if checkpoint_path is not None:
        target = Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_config": model_config.to_dict(),
                "training_config": asdict(training_config),
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "positive_class_weight": positive_weight,
            },
            target,
        )
        checkpoint_name = str(target)

    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        training_losses=training_losses,
        validation_losses=validation_losses,
        positive_class_weight=positive_weight,
        checkpoint_path=checkpoint_name,
    )


def load_model_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[MultimodalGRU, dict[str, object]]:
    """Load a model checkpoint containing only configuration and tensor state."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = MultimodalGRU(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict_patient(
    model: MultimodalGRU,
    sequence: PatientSequence,
    threshold: float = 0.5,
) -> PredictionResult:
    """Predict an anchor outcome and return descriptive prefix scores."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    device = next(model.parameters()).device
    batch = collate_patient_sequences([sequence]).to(device)
    model.eval()
    logits = model(batch)[0, : sequence.visit_count]
    prefix_probabilities = torch.sigmoid(logits).cpu().tolist()
    probability = float(prefix_probabilities[-1])
    return PredictionResult(
        subject_key=sequence.subject_key,
        probability=probability,
        threshold=threshold,
        predicted_label=int(probability >= threshold),
        prefix_probabilities=tuple(float(value) for value in prefix_probabilities),
    )


def predict_probabilities(
    model: MultimodalGRU,
    sequences: Sequence[PatientSequence],
    batch_size: int = 32,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return final anchor probabilities and valid prefix scores in input order."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not sequences:
        return np.empty(0, dtype=np.float64), []
    device = next(model.parameters()).device
    loader = _make_loader(sequences, batch_size, shuffle=False, seed=0)
    probabilities: list[float] = []
    prefixes: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            lengths = batch.sequence_lengths.cpu().tolist()
            for row, length in enumerate(lengths):
                values = torch.sigmoid(logits[row, :length]).cpu().numpy().astype(np.float64)
                prefixes.append(values)
                probabilities.append(float(values[-1]))
    return np.asarray(probabilities, dtype=np.float64), prefixes
