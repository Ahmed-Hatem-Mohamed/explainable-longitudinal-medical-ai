from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
import pytest

from workstream1.pipeline import (
    build_cohort,
    load_config,
    make_patient_splits,
    match_mri_to_clinical,
)


@pytest.fixture
def config(tmp_path):
    cfg = {
        "source_freeze": "SYNTHETIC_TEST_DATA",
        "clinical": {
            "participant_id": "NACCID", "site_id": "NACCADC", "visit_date": "VISITDATE",
            "diagnosis": "NACCUDSD", "alzheimers_etiology": "NACCALZD",
            "features": ["NACCAGE", "NACCSEX", "EDUC", "CDRGLOB", "CDRSUM", "NACCMMSE"]
        },
        "mri": {"participant_id": "NACCID", "site_id": "NACCADC", "year": "MRIYR", "month": "MRIMO", "day": "MRIDY"},
        "diagnosis_codes": {"mci": 3, "dementia": 4, "alzheimers_etiology": 1},
        "cohort": {"max_mri_clinical_delta_days": 90, "progression_window_days": 730,
                   "minimum_sequence_visits": 2, "maximum_sequence_visits": 3,
                   "train_fraction": .70, "validation_fraction": .15, "test_fraction": .15,
                   "random_seed": 42}
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def _make_data(n=30):
    clinical, mri = [], []
    for i in range(n):
        pid = f"SYN{i:03d}"
        base = pd.Timestamp("2020-01-01")
        # All participants have a prior matched observation and an MCI anchor.
        visits = [(0, 2), (365, 3)]
        if i % 2 == 0:
            visits.append((730, 4))  # positive within 730 days from anchor
        else:
            visits.append((1095, 3))  # sufficient follow-up without conversion in window
        for offset, diagnosis in visits:
            date = base + timedelta(days=int(offset))
            clinical.append({"NACCID": pid, "NACCADC": 1, "VISITDATE": date.strftime("%Y-%m-%d"),
                             "NACCUDSD": diagnosis, "NACCALZD": 1 if diagnosis == 4 else 0,
                             "NACCAGE": 70, "NACCSEX": 1, "EDUC": 16, "CDRGLOB": 0.5,
                             "CDRSUM": 2.0, "NACCMMSE": 27})
            mri.append({"NACCID": pid, "NACCADC": 1, "MRIYR": date.year, "MRIMO": date.month,
                        "MRIDY": date.day, "NACCMRIA": 1})
    return pd.DataFrame(clinical), pd.DataFrame(mri)


def test_load_config_rejects_unset_freeze(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"source_freeze":"SET_ME_BEFORE_RUNNING","cohort":{"train_fraction":0.7,"validation_fraction":0.15,"test_fraction":0.15}}')
    with pytest.raises(ValueError, match="source_freeze"):
        load_config(path)


def test_matching_and_cohort_have_one_anchor_per_participant(config):
    clinical, mri = _make_data()
    matched, audit = match_mri_to_clinical(clinical, mri, config)
    manifest, sequences = build_cohort(matched, config)
    assert audit["matched_mri_rows"] == len(matched)
    assert manifest["participant_id"].is_unique
    assert len(manifest) == 30
    assert set(manifest["label"].unique()) == {0, 1}
    assert sequences.groupby("clinical_NACCID").size().between(2, 3).all()


def test_splits_are_disjoint_and_cover_every_participant(config):
    manifest = pd.DataFrame({"participant_id": [f"p{i}" for i in range(40)], "label": [i % 2 for i in range(40)]})
    splits = make_patient_splits(manifest, config)
    sets = [set(splits[key]) for key in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(manifest["participant_id"])
