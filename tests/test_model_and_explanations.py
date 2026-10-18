import numpy as np
import torch

from nacc_progression.contracts import PatientSequence, collate_patient_sequences
from nacc_progression.explainability import (
    assess_feature_ablation_fidelity,
    assess_explanation_stability,
    explain_patient,
    leave_one_visit_out,
)
from nacc_progression.model import ModelConfig, MultimodalGRU, final_valid_logits
from nacc_progression.training import (
    TrainingConfig,
    load_model_checkpoint,
    predict_patient,
    train_model,
)
from tests.conftest import make_sequence


def test_packed_gru_is_invariant_to_batch_padding():
    short = make_sequence("short", 0, visits=1)
    long = make_sequence("long", 1, visits=3)
    model = MultimodalGRU(ModelConfig(imaging_features=4, dropout=0.0)).eval()

    mixed_batch = collate_patient_sequences([short, long])
    mixed_logits = model(mixed_batch)
    mixed_final = final_valid_logits(mixed_logits, mixed_batch.sequence_mask)
    short_alone = model(collate_patient_sequences([short]))
    long_alone = model(collate_patient_sequences([long]))

    assert mixed_logits.shape == torch.Size([2, 3])
    torch.testing.assert_close(mixed_final[0], short_alone[0, 0])
    torch.testing.assert_close(mixed_final[1], long_alone[0, 2])
    assert mixed_batch.sequence_mask.tolist() == [[True, False, False], [True, True, True]]


def test_single_modality_longitudinal_models_run():
    sequence = make_sequence("clinical-only", 1, visits=2)
    batch = collate_patient_sequences([sequence])
    for config in (
        ModelConfig(imaging_features=4, use_imaging=False),
        ModelConfig(imaging_features=4, use_clinical=False),
    ):
        model = MultimodalGRU(config).eval()
        assert model(batch).shape == (1, 2)


def test_sequence_rejects_invalid_time_and_mask_shapes():
    sequence = make_sequence("valid", 0)
    try:
        PatientSequence(
            subject_key="invalid-time",
            label=0,
            imaging_values=sequence.imaging_values,
            imaging_missing_mask=sequence.imaging_missing_mask,
            clinical_values=sequence.clinical_values,
            clinical_missing_mask=sequence.clinical_missing_mask,
            delta_time=np.asarray([[0.5], [0.5]], dtype=np.float32),
        )
    except ValueError as error:
        assert "start at zero" in str(error)
    else:
        raise AssertionError("invalid time origin was accepted")


def test_training_checkpoint_prediction_and_explanation_roundtrip(tmp_path):
    training = [
        make_sequence("train-0", 0),
        make_sequence("train-1", 1),
        make_sequence("train-2", 0, visits=3),
        make_sequence("train-3", 1, visits=1),
    ]
    validation = [make_sequence("validation-0", 0), make_sequence("validation-1", 1)]
    checkpoint = tmp_path / "model.pt"
    result = train_model(
        training,
        validation,
        ModelConfig(imaging_features=4, dropout=0.0),
        TrainingConfig(max_epochs=2, patience=1, batch_size=2, seed=7),
        checkpoint_path=checkpoint,
    )
    loaded_model, metadata = load_model_checkpoint(checkpoint)
    prediction = predict_patient(loaded_model, validation[0])

    assert result.best_epoch >= 1
    assert metadata["format_version"] == 1
    assert 0 <= prediction.probability <= 1
    assert len(prediction.prefix_probabilities) == validation[0].visit_count

    explanation = explain_patient(loaded_model, validation[0], steps=16)
    assert explanation.mri_attributions.shape == (2, 4)
    assert explanation.clinical_attributions.shape == (2, 6)
    assert explanation.delta_time_attributions.shape == (2, 1)
    assert np.isfinite(explanation.completeness_delta_logit)
    assert np.isclose(sum(explanation.modality_importance.values()), explanation.visit_importance.sum())
    assert len(leave_one_visit_out(loaded_model, validation[0])) == 2
    fidelity = assess_feature_ablation_fidelity(
        loaded_model, validation[0], explanation=explanation, top_k=3
    )
    assert len(fidelity["mri_regional_features"]["features"]) == 4
    assert len(fidelity["clinical_variables"]["features"]) == 6

    stability = assess_explanation_stability(
        loaded_model,
        validation[0],
        perturbation_replicates=2,
        integrated_gradients_steps=8,
        top_k=3,
    )
    assert stability["perturbation_replicates"] == 2
    assert 0 <= stability["mean_perturbation_top_k_jaccard"] <= 1
