import numpy as np
import pytest

from nacc_progression.baselines import last_visit_features
from nacc_progression.evaluation import (
    bootstrap_confidence_intervals,
    calibration_curve_data,
    calculate_metrics,
    expected_calibration_error,
    run_model_comparison,
    select_balanced_accuracy_threshold,
)
from nacc_progression.model import ModelConfig
from nacc_progression.training import TrainingConfig
from tests.conftest import make_sequence


def test_last_visit_baselines_are_non_longitudinal_and_keep_masks():
    sequences = [make_sequence("patient-0", 0, visits=1), make_sequence("patient-1", 1, visits=3)]
    clinical = last_visit_features(sequences, "clinical")
    imaging = last_visit_features(sequences, "imaging")
    multimodal = last_visit_features(sequences, "multimodal")

    assert clinical.shape == (2, 12)
    assert imaging.shape == (2, 8)
    assert multimodal.shape == (2, 20)
    np.testing.assert_array_equal(multimodal[:, :8], imaging)
    np.testing.assert_array_equal(multimodal[:, 8:], clinical)
    np.testing.assert_array_equal(clinical[1, :6], sequences[1].clinical_values[-1])


def test_threshold_is_selected_on_validation_balanced_accuracy():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9])

    threshold = select_balanced_accuracy_threshold(labels, probabilities)

    assert threshold == 0.5
    report = calculate_metrics(labels, probabilities, threshold)
    assert report["auroc"] == 1.0
    assert report["auprc"] == 1.0
    assert report["balanced_accuracy"] == 1.0
    assert report["sensitivity"] == 1.0
    assert report["specificity"] == 1.0
    assert report["brier_score"] < 0.1


def test_calibration_and_patient_cluster_bootstrap():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]
    assert np.isclose(
        expected_calibration_error(labels, probabilities, bin_count=2), 0.15
    )

    intervals = bootstrap_confidence_intervals(
        labels,
        probabilities,
        ["p0", "p1", "p2", "p3"],
        threshold=0.5,
        repetitions=50,
        seed=3,
    )
    assert set(intervals) == {"auroc", "auprc", "balanced_accuracy", "brier_score"}
    assert intervals["auroc"] == (1.0, 1.0)
    curve = calibration_curve_data(labels, probabilities, bin_count=2)
    assert curve["counts"] == [2.0, 2.0]
    assert np.isclose(curve["mean_predicted_probability"][0], 0.15)
    assert np.isclose(curve["observed_event_rate"][1], 1.0)


def test_comparison_runs_all_six_fixed_arms():
    training = [make_sequence(f"train-{index}", index % 2) for index in range(4)]
    validation = [make_sequence(f"validation-{index}", index % 2) for index in range(4)]
    test = [make_sequence(f"test-{index}", index % 2) for index in range(4)]

    results = run_model_comparison(
        training,
        validation,
        test,
        ModelConfig(imaging_features=4, dropout=0.0),
        TrainingConfig(max_epochs=1, patience=1, batch_size=4, seed=9),
        bootstrap_repetitions=10,
    )

    assert len(results) == 6
    assert "logistic_clinical_last_visit" in results
    assert "logistic_imaging_last_visit" in results
    assert "logistic_multimodal_last_visit" in results
    assert "gru_clinical_longitudinal" in results
    assert "gru_imaging_longitudinal" in results
    assert "gru_multimodal_longitudinal" in results
    assert all(result.test_metrics["n"] == 4 for result in results.values())


def test_comparison_rejects_mri_feature_order_mismatch_before_training():
    expected = ("hippocampus", "entorhinal", "ventricle", "amygdala")
    reversed_order = tuple(reversed(expected))
    training = [make_sequence("train-0", 0, imaging_feature_names=expected)]
    validation = [make_sequence("validation-0", 1, imaging_feature_names=reversed_order)]
    test = [make_sequence("test-0", 0, imaging_feature_names=expected)]

    with pytest.raises(ValueError, match="ordered MRI feature names"):
        run_model_comparison(
            training,
            validation,
            test,
            ModelConfig(imaging_features=4, imaging_feature_names=expected),
            bootstrap_repetitions=1,
        )
