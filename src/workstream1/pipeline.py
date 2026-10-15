"""Cohort construction utilities for longitudinal NACC clinical/MRI data.

This module intentionally reads participant-level data only from user-specified local paths.
It never downloads data and never writes participant-level outputs into the Git repository by
itself; choose an output directory covered by .gitignore (normally data/processed/).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the JSON pipeline configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("source_freeze") in (None, "", "SET_ME_BEFORE_RUNNING"):
        raise ValueError("Set config.source_freeze to the actual source freeze before running.")
    cohort = config["cohort"]
    fractions = [cohort["train_fraction"], cohort["validation_fraction"], cohort["test_fraction"]]
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")
    if cohort["minimum_sequence_visits"] < 2:
        raise ValueError("minimum_sequence_visits must be at least 2.")
    if cohort["maximum_sequence_visits"] < cohort["minimum_sequence_visits"]:
        raise ValueError("maximum_sequence_visits must be >= minimum_sequence_visits.")
    return config


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _as_date(series: pd.Series) -> pd.Series:
    """Parse dates conservatively; invalid or incomplete dates become NaT."""
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _mri_date(mri: pd.DataFrame, mri_cfg: dict[str, Any]) -> pd.Series:
    year_col, month_col, day_col = mri_cfg["year"], mri_cfg["month"], mri_cfg["day"]
    _require_columns(mri, [year_col, month_col, day_col], "MRI input")
    return pd.to_datetime(
        {
            "year": pd.to_numeric(mri[year_col], errors="coerce"),
            "month": pd.to_numeric(mri[month_col], errors="coerce"),
            "day": pd.to_numeric(mri[day_col], errors="coerce"),
        },
        errors="coerce",
    ).dt.normalize()


def match_mri_to_clinical(
    clinical: pd.DataFrame, mri: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Match each MRI record to the nearest same-participant clinical visit within a day limit.

    Matching is done by absolute date difference. A clinical visit can be selected for more than
    one MRI record if it is independently the nearest visit; this behavior is explicit and audited.
    """
    cc, mc = config["clinical"], config["mri"]
    c_required = [cc[k] for k in ("participant_id", "site_id", "visit_date", "diagnosis", "alzheimers_etiology")]
    m_required = [mc[k] for k in ("participant_id", "site_id")]
    _require_columns(clinical, c_required + list(cc["features"]), "Clinical input")
    _require_columns(mri, m_required + [mc["year"], mc["month"], mc["day"]], "MRI input")

    clin = clinical.copy().reset_index(drop=True)
    scans = mri.copy().reset_index(drop=True)
    clin["_clinical_date"] = _as_date(clin[cc["visit_date"]])
    scans["_mri_date"] = _mri_date(scans, mc)
    clin = clin.loc[clin["_clinical_date"].notna()].copy()
    scans = scans.loc[scans["_mri_date"].notna()].copy()
    clin["_participant"] = clin[cc["participant_id"]].astype("string")
    scans["_participant"] = scans[mc["participant_id"]].astype("string")
    clin["_site"] = clin[cc["site_id"]].astype("string")
    scans["_site"] = scans[mc["site_id"]].astype("string")

    max_delta = int(config["cohort"]["max_mri_clinical_delta_days"])
    matches: list[dict[str, Any]] = []
    # Grouping reduces unnecessary comparisons while retaining deterministic tie-breaking.
    clin_groups = {
        key: group.sort_values(["_clinical_date"], kind="stable")
        for key, group in clin.groupby(["_participant", "_site"], dropna=False, sort=False)
    }
    for _, scan in scans.iterrows():
        key = (scan["_participant"], scan["_site"])
        candidates = clin_groups.get(key)
        if candidates is None or candidates.empty:
            continue
        delta = (candidates["_clinical_date"] - scan["_mri_date"]).dt.days
                # Prediction is anchored at the clinical assessment: never use a later MRI.
        eligible = delta.loc[delta.between(0, max_delta)]
        if eligible.empty:
            continue

        min_delta = eligible.min()
        chosen_idx = candidates.loc[eligible.index[eligible.eq(min_delta)]].index[0]
        visit = clin.loc[chosen_idx]
        row = {column: scan[column] for column in mri.columns}
        row.update({f"clinical_{column}": visit[column] for column in clinical.columns})
        row["mri_date"] = scan["_mri_date"]
        row["visit_date"] = visit["_clinical_date"]
        row["delta_days"] = int((visit["_clinical_date"] - scan["_mri_date"]).days)
        matches.append(row)

    matched = pd.DataFrame(matches)
    if matched.empty:
        raise ValueError("No MRI/clinical matches found. Check dates, participant/site keys, and config.")
    audit: dict[str, int | float] = {
        "clinical_rows_with_valid_date": len(clin),
        "mri_rows_with_valid_date": len(scans),
        "matched_mri_rows": len(matched),
        "matched_participants": int(matched[f"clinical_{cc['participant_id']}"].nunique()),
        "median_absolute_delta_days": float(matched["delta_days"].abs().median()),
        "p95_absolute_delta_days": float(matched["delta_days"].abs().quantile(0.95)),
        "max_absolute_delta_days": int(matched["delta_days"].abs().max()),
    }
    return matched, audit


def build_cohort(matched: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one anchor and a 2–3 visit sequence per eligible participant.

    Positive: MCI anchor followed by AD dementia with AD etiology within the configured window.
    Negative: no qualifying AD dementia within the window and at least a full window of observed
    clinical follow-up after the anchor. Participants with insufficient follow-up are excluded.
    """
    cc, cohort, codes = config["clinical"], config["cohort"], config["diagnosis_codes"]
    pid = f"clinical_{cc['participant_id']}"
    diagnosis = f"clinical_{cc['diagnosis']}"
    etiology = f"clinical_{cc['alzheimers_etiology']}"
    visit_date = "visit_date"
    required = [pid, diagnosis, etiology, visit_date, "mri_date", "delta_days"] + [
        f"clinical_{feature}" for feature in cc["features"]
    ]
    _require_columns(matched, required, "Matched data")

    data = matched.copy()
    data[pid] = data[pid].astype("string")
    data[visit_date] = _as_date(data[visit_date])
    data = data.loc[data[visit_date].notna()].copy()
    data = data.sort_values([pid, visit_date, "mri_date"], kind="stable")
    window = int(cohort["progression_window_days"])
    min_visits, max_visits = int(cohort["minimum_sequence_visits"]), int(cohort["maximum_sequence_visits"])
    manifest_rows: list[dict[str, Any]] = []
    sequence_rows: list[pd.DataFrame] = []

    for subject_id, group in data.groupby(pid, sort=False, dropna=False):
        group = group.sort_values([visit_date, "mri_date"], kind="stable").drop_duplicates(
            subset=[visit_date], keep="first"
        )
        if len(group) < min_visits:
            continue
        diagnoses = pd.to_numeric(group[diagnosis], errors="coerce")
        etiologies = pd.to_numeric(group[etiology], errors="coerce")
        group = group.assign(_diagnosis=diagnoses, _etiology=etiologies)
        group_dates = group[visit_date]
        latest_followup = group_dates.max()
        anchors = group.loc[group["_diagnosis"].eq(codes["mci"])].copy()
        eligible_for_subject: list[tuple[pd.Series, int, int]] = []
        for anchor_idx, anchor in anchors.iterrows():
            anchor_date = anchor[visit_date]
            prior = group.loc[group[visit_date] < anchor_date]
            if prior.empty:
                continue
            future = group.loc[group[visit_date] > anchor_date]
            future_delta = (future[visit_date] - anchor_date).dt.days
            is_ad = future["_diagnosis"].eq(codes["dementia"]) & future["_etiology"].eq(
                codes["alzheimers_etiology"]
            )
            days_to_ad_values = future_delta.loc[is_ad & future_delta.le(window)]
            if not days_to_ad_values.empty:
                label = 1
                days_to_ad = int(days_to_ad_values.min())
            else:
                followup_days = int((latest_followup - anchor_date).days)
                if followup_days < window:
                    continue
                label = 0
                days_to_ad = -1
            eligible_for_subject.append((anchor, label, days_to_ad))

        if not eligible_for_subject:
            continue
        # Choose latest eligible MCI anchor; this prevents multiple rows per participant.
        anchor, label, days_to_ad = max(eligible_for_subject, key=lambda item: item[0][visit_date])
        anchor_date = anchor[visit_date]
        past = group.loc[group[visit_date].le(anchor_date)].tail(max_visits).copy()
        if len(past) < min_visits:
            continue
        past["label"] = int(label)
        past["anchor_date"] = anchor_date
        past["delta_time_days"] = (past[visit_date] - anchor_date).dt.days
        past["is_anchor"] = past[visit_date].eq(anchor_date).astype(int)
        past["sequence_length"] = len(past)
        sequence_rows.append(past.drop(columns=["_diagnosis", "_etiology"], errors="ignore"))
        manifest_rows.append({
            "participant_id": str(subject_id),
            "anchor_date": anchor_date,
            "label": int(label),
            "days_to_ad": int(days_to_ad),
            "followup_days": int((latest_followup - anchor_date).days),
            "sequence_length": len(past),
            "prior_count": int((group[visit_date] < anchor_date).sum()),
        })

    manifest = pd.DataFrame(manifest_rows)
    sequences = pd.concat(sequence_rows, ignore_index=True) if sequence_rows else pd.DataFrame()
    if manifest.empty:
        raise ValueError("No eligible participants after label and sequence criteria.")
    if manifest["participant_id"].duplicated().any():
        raise AssertionError("Cohort manifest contains duplicate participants.")
    return manifest, sequences


def make_patient_splits(manifest: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[str]]:
    """Create stratified patient-level train/validation/test splits with disjoint IDs."""
    required = ["participant_id", "label"]
    _require_columns(manifest, required, "Cohort manifest")
    if manifest["participant_id"].duplicated().any():
        raise ValueError("Manifest must contain one row per participant before splitting.")
    labels = manifest["label"].astype(int)
    ids = manifest["participant_id"].astype(str)
    seed = int(config["cohort"]["random_seed"])
    train_frac = float(config["cohort"]["train_fraction"])
    val_frac = float(config["cohort"]["validation_fraction"])
    test_frac = float(config["cohort"]["test_fraction"])
    if labels.value_counts().min() < 3:
        raise ValueError("Need at least 3 participants in each class for stratified three-way split.")
    train_ids, rest_ids, _, rest_y = train_test_split(
        ids, labels, train_size=train_frac, random_state=seed, stratify=labels
    )
    relative_test = test_frac / (val_frac + test_frac)
    val_ids, test_ids = train_test_split(
        rest_ids,
        test_size=relative_test,
        random_state=seed,
        stratify=rest_y,
    )
    splits = {
        "train": sorted(train_ids.astype(str).tolist()),
        "validation": sorted(val_ids.astype(str).tolist()),
        "test": sorted(test_ids.astype(str).tolist()),
    }
    sets = [set(v) for v in splits.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Patient leakage detected between splits.")
    if set.union(*sets) != set(ids):
        raise AssertionError("Split membership does not cover the complete cohort.")
    return splits


def write_outputs(
    manifest: pd.DataFrame,
    sequences: pd.DataFrame,
    matched_audit: dict[str, Any],
    splits: dict[str, list[str]],
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write local CSV/JSON outputs and a privacy-conscious aggregate audit."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "cohort_manifest.csv", index=False)
    sequences.to_csv(out / "visits_clinical.csv", index=False)
    with (out / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)

    split_map = {participant: name for name, participants in splits.items() for participant in participants}
    labeled = manifest.copy()
    labeled["split"] = labeled["participant_id"].astype(str).map(split_map)
    audit = {
        "source_freeze": config["source_freeze"],
        "matched_data": matched_audit,
        "eligible_participants": len(manifest),
        "label_counts": {str(int(k)): int(v) for k, v in manifest["label"].value_counts().sort_index().items()},
        "sequence_length_counts": {str(int(k)): int(v) for k, v in manifest["sequence_length"].value_counts().sort_index().items()},
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "split_label_counts": {
            name: {str(int(k)): int(v) for k, v in labeled.loc[labeled["split"].eq(name), "label"].value_counts().sort_index().items()}
            for name in splits
        },
        "patient_leakage": 0,
        "clinical_feature_missing_rates": {
            column.removeprefix("clinical_"): float(sequences[column].isna().mean())
            for column in sequences.columns
            if column.startswith("clinical_") and column.removeprefix("clinical_") in config["clinical"]["features"]
        },
        "warning": "Participant-level outputs are restricted and must not be committed to Git.",
    }
    with (out / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=str)
    return audit

