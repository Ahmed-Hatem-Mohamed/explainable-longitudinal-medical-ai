"""Multimodal visit encoder and packed-GRU progression model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nacc_progression.contracts import PatientBatch


@dataclass(frozen=True)
class ModelConfig:
    imaging_features: int
    clinical_features: int = 6
    imaging_embedding: int = 64
    clinical_embedding: int = 32
    temporal_hidden: int = 64
    dropout: float = 0.20
    use_imaging: bool = True
    use_clinical: bool = True
    imaging_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.imaging_features < 1:
            raise ValueError("imaging_features must be positive")
        if min(self.imaging_embedding, self.clinical_embedding, self.temporal_hidden) < 1:
            raise ValueError("embedding and hidden sizes must be positive")
        if self.clinical_features != 6:
            raise ValueError("NACC V1 expects six clinical variables per visit")
        names = tuple(self.imaging_feature_names)
        object.__setattr__(self, "imaging_feature_names", names)
        if names and len(names) != self.imaging_features:
            raise ValueError("imaging_feature_names must match imaging_features")
        if len(set(names)) != len(names):
            raise ValueError("imaging_feature_names must be unique")
        if not self.use_imaging and not self.use_clinical:
            raise ValueError("at least one modality must be enabled")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ModelConfig:
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MultimodalGRU(nn.Module):
    """Predict the anchor outcome from a patient history of up to three visits.

    ``forward`` returns one logit per valid visit prefix. Training should use only
    the final valid logit with the anchor label; earlier prefix outputs are
    descriptive scores for that same anchor outcome, not visit-specific 24-month
    targets. Modality flags support longitudinal unimodal comparison models.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_imaging:
            self.imaging_encoder = nn.Sequential(
                nn.Linear(2 * config.imaging_features, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, config.imaging_embedding),
                nn.ReLU(),
            )
        else:
            self.imaging_encoder = None

        if config.use_clinical:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(2 * config.clinical_features, 32),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(32, config.clinical_embedding),
                nn.ReLU(),
            )
        else:
            self.clinical_encoder = None

        visit_width = 1
        if config.use_imaging:
            visit_width += config.imaging_embedding
        if config.use_clinical:
            visit_width += config.clinical_embedding
        self.temporal_encoder = nn.GRU(
            input_size=visit_width,
            hidden_size=config.temporal_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(config.temporal_hidden, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
        )

    def forward(self, batch: PatientBatch) -> Tensor:
        """Return logits shaped [batch, visits], with padding positions set to zero."""

        if batch.imaging_values.shape[-1] != self.config.imaging_features:
            raise ValueError("batch MRI feature count does not match model configuration")
        if batch.clinical_values.shape[-1] != self.config.clinical_features:
            raise ValueError("batch clinical feature count does not match model configuration")
        if not batch.sequence_mask.any(dim=1).all():
            raise ValueError("every sequence must contain at least one valid visit")

        visit_embeddings = []
        if self.imaging_encoder is not None:
            imaging_input = torch.cat(
                (
                    batch.imaging_values,
                    batch.imaging_missing_mask.to(dtype=batch.imaging_values.dtype),
                ),
                dim=-1,
            )
            visit_embeddings.append(self.imaging_encoder(imaging_input))
        if self.clinical_encoder is not None:
            clinical_input = torch.cat(
                (
                    batch.clinical_values,
                    batch.clinical_missing_mask.to(dtype=batch.clinical_values.dtype),
                ),
                dim=-1,
            )
            visit_embeddings.append(self.clinical_encoder(clinical_input))
        visit_embeddings.append(batch.delta_time.to(dtype=batch.imaging_values.dtype))
        visits = torch.cat(visit_embeddings, dim=-1)

        lengths = batch.sequence_lengths
        packed = pack_padded_sequence(
            visits,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.temporal_encoder(packed)
        temporal_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=visits.shape[1],
        )
        logits = self.prediction_head(temporal_output).squeeze(-1)
        return logits.masked_fill(~batch.sequence_mask, 0.0)


def final_valid_logits(logits: Tensor, sequence_mask: Tensor) -> Tensor:
    """Select the anchor logit from each padded batch row."""

    lengths = sequence_mask.sum(dim=1).to(dtype=torch.long)
    if torch.any(lengths < 1):
        raise ValueError("every sequence must contain at least one valid visit")
    row_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[row_indices, lengths.to(device=logits.device) - 1]