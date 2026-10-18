import numpy as np

from nacc_progression.contracts import PatientSequence


def make_sequence(
    subject_key: str,
    label: int,
    visits: int = 2,
    mri_features: int = 4,
    imaging_feature_names: tuple[str, ...] = (),
):
    rng = np.random.default_rng(sum(ord(character) for character in subject_key))
    return PatientSequence(
        subject_key=subject_key,
        label=label,
        imaging_values=rng.normal(size=(visits, mri_features)).astype(np.float32),
        imaging_missing_mask=np.zeros((visits, mri_features), dtype=bool),
        clinical_values=rng.normal(size=(visits, 6)).astype(np.float32),
        clinical_missing_mask=np.zeros((visits, 6), dtype=bool),
        delta_time=np.asarray([[0.0]] + [[0.5] for _ in range(visits - 1)], dtype=np.float32),
        imaging_feature_names=imaging_feature_names,
    )
