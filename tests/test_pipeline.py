from __future__ import annotations

import importlib.util
import json
from datetime import timedelta

import pandas as pd
import pytest

from workstream1.pipeline import (
    _fit_training_preprocessor,
    _safe_output_frames,
    build_cohort,
    load_config,
    make_patient_splits,
    match_mri_to_clinical,
    write_outputs,
)


@pytest.fixture
def config(tmp_path):
    cfg = {
        "source_freeze": "SYNTHETIC_TEST_DATA",
        "clinical": {
            "participant_id": "NACCID",
            "site_id": "NACCADC",
            "visit_date": "VISITDATE",
            "diagnosis": "NACCUDSD",
            "alzheimers_etiology": "NACCALZD",
            "features": ["NACCAGE", "NACCSEX", "EDUC", "CDRGLOB", "CDRSUM", "NACCMMSE"],
        },
        "mri": {
            "participant_id": "NACCID", "site_id": "NACCADC",
            "year": "MRIYR", "month": "MRIMO", "day": "MRIDY",
        },
        "diagnosis_codes": {"mci": 3, "dementia": 4, "alzheimers_etiology": 1},
        "feature_missing_codes": {},
        "categorical_features": ["NACCSEX"],
        "cohort": {
            "max_mri_clinical_delta_days": 90,
            "progression_window_days": 730,
            "minimum_sequence_visits": 2,
            "maximum_sequence_visits": 3,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "random_seed": 42,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def _make_data(n=30):
    clinical, mri = [], []
    for i in range(n):
        pid = f"SYN{i:03d}"
        base = pd.Timestamp("2020-01-01")
        visits = [(0, 2), (365, 3)]
        if i % 2 == 0:
            visits.append((730, 4))
        else:
            visits.append((1095, 3))
        for offset, diagnosis in visits:
            date = base + timedelta(days=int(offset))
            clinical.append({
                "NACCID": pid, "NACCADC": 1, "VISITDATE": date.strftime("%Y-%m-%d"),
                "NACCUDSD": diagnosis, "NACCALZD": 1 if diagnosis == 4 else 0,
                "NACCAGE": 70 + i % 5, "NACCSEX": 1 + i % 2, "EDUC": 16,
                "CDRGLOB": 0.5, "CDRSUM": 2.0, "NACCMMSE": 27,
            })
            mri.append({
                "NACCID": pid, "NACCADC": 1, "MRIYR": date.year,
                "MRIMO": date.month, "MRIDY": date.day, "NACCMRIA": 1,
            })
    return pd.DataFrame(clinical), pd.DataFrame(mri)


def test_load_config_rejects_unset_freeze(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"source_freeze":"SET_ME_BEFORE_RUNNING","clinical":{},"mri":{},'
        '"diagnosis_codes":{},"cohort":{}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source_freeze"):
        load_config(path)


def test_matching_and_cohort_use_full_clinical_timeline(config):
    clinical, mri = _make_data()
    matched, match_audit = match_mri_to_clinical(clinical, mri, config)
    manifest, sequences, cohort_audit = build_cohort(matched, clinical, config)

    assert match_audit["matched_mri_rows"] == len(matched)
    assert match_audit["future_mri_matches"] == 0
    assert manifest["participant_id"].is_unique
    assert len(manifest) == 30
    assert set(manifest["label"].unique()) == {0, 1}
    assert sequences.groupby("_join_key").size().between(2, 3).all()
    assert sequences.groupby("_join_key")["is_anchor"].sum().eq(1).all()
    assert (sequences["delta_time_days"] >= 0).all()
    assert cohort_audit["eligible_participants"] == 30


def test_future_mri_is_not_matched(config):
    clinical, mri = _make_data(n=1)
    mri["MRIYR"] = 2020
    mri["MRIMO"] = 1
    mri["MRIDY"] = 2
    # A scan after the only clinical visit must not be matched to that visit.
    with pytest.raises(ValueError, match="No MRI/clinical matches"):
        match_mri_to_clinical(clinical.iloc[[0]], mri.iloc[[0]], config)


def test_negative_requires_full_followup_and_competing_dementia_is_excluded(config):
    clinical, mri = _make_data(n=4)
    # Participant 1 has insufficient full clinical follow-up after MCI; remove its last visit.
    clinical = clinical.loc[~((clinical["NACCID"] == "SYN001") & (clinical["VISITDATE"] == "2022-12-31"))].copy()
    mri = mri.loc[~((mri["NACCID"] == "SYN001") & (mri["MRIYR"] == 2022))].copy()
    # Participant 2 develops dementia with non-AD etiology: this is a competing event, not a negative.
    clinical.loc[(clinical["NACCID"] == "SYN002") & (clinical["NACCUDSD"] == 4), "NACCALZD"] = 2
    matched, _ = match_mri_to_clinical(clinical, mri, config)
    manifest, _, audit = build_cohort(matched, clinical, config)
    assert "\x1fSYN001" not in set(manifest["participant_id"])
    assert "\x1fSYN002" not in set(manifest["participant_id"])
    assert audit["exclusions"]["participants_with_insufficient_followup"] >= 1
    assert audit["exclusions"]["participants_with_competing_or_ambiguous_dementia"] >= 1


def test_outcomes_use_full_clinical_timeline_even_without_mri(config):
    clinical, mri = _make_data(n=4)
    # Keep the clinical AD event for SYN000 but remove its MRI; the event must still label progression.
    mri = mri.loc[~((mri["NACCID"] == "SYN000") & (mri["MRIYR"] == 2021))].copy()
    # Keep SYN001's final clinical follow-up but remove its MRI; it can still establish a negative.
    mri = mri.loc[~((mri["NACCID"] == "SYN001") & (mri["MRIYR"] == 2022))].copy()
    matched, _ = match_mri_to_clinical(clinical, mri, config)
    manifest, _, _ = build_cohort(matched, clinical, config)
    labels = manifest.set_index("participant_id")["label"]
    assert labels.loc[labels.index.to_series().str.endswith("SYN000").idxmax()] == 1
    assert labels.loc[labels.index.to_series().str.endswith("SYN001").idxmax()] == 0


def test_splits_are_disjoint_and_cover_every_participant(config):
    manifest = pd.DataFrame({"participant_id": [f"p{i}" for i in range(40)], "label": [i % 2 for i in range(40)]})
    splits = make_patient_splits(manifest, config)
    sets = [set(splits[key]) for key in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(manifest["participant_id"])


def test_pseudonymization_and_preprocessing_are_training_scoped(config):
    clinical, mri = _make_data()
    matched, _ = match_mri_to_clinical(clinical, mri, config)
    manifest, sequences, _ = build_cohort(matched, clinical, config)
    splits = make_patient_splits(manifest, config)
    manifest_out, visits_out, splits_out, key_to_subject = _safe_output_frames(
        manifest, sequences, splits, config, "synthetic-test-secret-that-is-long-enough"
    )
    model_ready, preprocessing = _fit_training_preprocessor(sequences, manifest, splits, config)
    model_ready["subject_key"] = model_ready["_join_key"].astype(str).map(key_to_subject)
    model_ready = model_ready.drop(columns=["_join_key"])

    assert "subject_key" in manifest_out and "participant_id" not in manifest_out
    assert "NACCID" not in visits_out and "clinical_NACCID" not in visits_out
    assert not visits_out.astype(str).apply(lambda col: col.str.contains("SYN000")).any().any()
    assert set(splits_out["train"] + splits_out["validation"] + splits_out["test"]) == set(manifest_out["subject_key"])
    assert "NACCAGE__scaled" in model_ready
    assert "NACCSEX__category_1" in model_ready
    assert preprocessing["fit_scope"] == "training participants only"


def test_output_writer_creates_parquet_json_and_audit_figures(config, tmp_path):
    if importlib.util.find_spec("pyarrow") is None:
        pytest.skip("pyarrow is required to exercise Parquet output; it is a project dependency")
    clinical, mri = _make_data()
    matched, match_audit = match_mri_to_clinical(clinical, mri, config)
    manifest, sequences, cohort_audit = build_cohort(matched, clinical, config)
    splits = make_patient_splits(manifest, config)
    audit = write_outputs(
        manifest, sequences, match_audit, cohort_audit, splits, config, tmp_path / "processed",
        identifier_secret="synthetic-test-secret-that-is-long-enough",
    )
    output = tmp_path / "processed"
    manifest_out = pd.read_parquet(output / "cohort_manifest.parquet")
    visits_out = pd.read_parquet(output / "visits_clinical.parquet")
    ready_out = pd.read_parquet(output / "visits_clinical_model_ready.parquet")
    split_json = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    assert "subject_key" in manifest_out and "participant_id" not in manifest_out
    assert "NACCID" not in visits_out and "clinical_NACCID" not in visits_out
    assert not visits_out.astype(str).apply(lambda col: col.str.contains("SYN000")).any().any()
    assert set(split_json["train"] + split_json["validation"] + split_json["test"]) == set(manifest_out["subject_key"])
    assert "NACCAGE__scaled" in ready_out
    assert "NACCSEX__category_1" in ready_out
    assert audit["patient_leakage"] == 0
    assert (output / "audit_label_counts.png").exists()
    assert (output / "audit_split_counts.png").exists()


def test_output_writer_requires_a_long_secret(config, tmp_path):
    clinical, mri = _make_data()
    matched, match_audit = match_mri_to_clinical(clinical, mri, config)
    manifest, sequences, cohort_audit = build_cohort(matched, clinical, config)
    splits = make_patient_splits(manifest, config)
    with pytest.raises(ValueError, match="WORKSTREAM1_ID_SECRET"):
        write_outputs(manifest, sequences, match_audit, cohort_audit, splits, config, tmp_path / "out", identifier_secret="short")
