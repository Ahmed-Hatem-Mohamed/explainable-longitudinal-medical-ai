"""Auditable longitudinal clinical/MRI cohort construction for Workstream 1.

Source records are read from caller-supplied local files. Participant identifiers are used only
in memory for joins and splits; write_outputs replaces them with keyed pseudonyms before writing.
The resulting files are still restricted participant-level research data and must not be committed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_PLACEHOLDERS = {None, "", "SET_ME_BEFORE_RUNNING", "UNKNOWN", "TBD"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the JSON configuration and fail early on incomplete/unsafe settings."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required_top = {"source_freeze", "clinical", "mri", "diagnosis_codes", "cohort"}
    missing_top = sorted(required_top - set(config))
    if missing_top:
        raise ValueError(f"Config is missing required sections: {missing_top}")
    if config["source_freeze"] in _PLACEHOLDERS:
        raise ValueError("Set config.source_freeze to the actual source freeze before running.")

    clinical = config["clinical"]
    mri = config["mri"]
    cohort = config["cohort"]
    diagnosis_codes = config["diagnosis_codes"]
    for key in ("participant_id", "site_id", "visit_date", "diagnosis", "alzheimers_etiology", "features"):
        if key not in clinical:
            raise ValueError(f"Config clinical section is missing {key!r}.")
    for key in ("participant_id", "site_id", "year", "month", "day"):
        if key not in mri:
            raise ValueError(f"Config mri section is missing {key!r}.")
    if not clinical["features"] or len(set(clinical["features"])) != len(clinical["features"]):
        raise ValueError("clinical.features must contain unique column names.")
    for key in ("mci", "dementia", "alzheimers_etiology"):
        if key not in diagnosis_codes:
            raise ValueError(f"diagnosis_codes is missing {key!r}.")
    required_cohort = (
        "max_mri_clinical_delta_days", "progression_window_days", "minimum_sequence_visits",
        "maximum_sequence_visits", "train_fraction", "validation_fraction", "test_fraction",
        "random_seed",
    )
    for key in required_cohort:
        if key not in cohort:
            raise ValueError(f"cohort is missing {key!r}.")

    fractions = [float(cohort[k]) for k in ("train_fraction", "validation_fraction", "test_fraction")]
    if any(not 0 < fraction < 1 for fraction in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train, validation, and test fractions must each be in (0, 1) and sum to 1.0.")
    if int(cohort["max_mri_clinical_delta_days"]) < 0:
        raise ValueError("max_mri_clinical_delta_days must be non-negative.")
    if int(cohort["progression_window_days"]) <= 0:
        raise ValueError("progression_window_days must be positive.")
    if int(cohort["minimum_sequence_visits"]) < 2:
        raise ValueError("minimum_sequence_visits must be at least 2.")
    if int(cohort["maximum_sequence_visits"]) < int(cohort["minimum_sequence_visits"]):
        raise ValueError("maximum_sequence_visits must be >= minimum_sequence_visits.")
    if clinical["participant_id"] == clinical["site_id"]:
        raise ValueError("clinical participant_id and site_id columns must differ.")
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


def _normalise_identifier(series: pd.Series) -> pd.Series:
    """Normalize whitespace and integer-like CSV floats without changing textual identifiers."""
    values = series.astype("string").str.strip()
    return values.str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True)


def _key_series(frame: pd.DataFrame, participant_col: str, site_col: str) -> pd.Series:
    """Create an internal composite key; missing keys are kept missing, never joined together."""
    participant = _normalise_identifier(frame[participant_col])
    site = _normalise_identifier(frame[site_col])
    valid = participant.notna() & participant.ne("") & site.notna() & site.ne("")
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    result.loc[valid] = site.loc[valid] + "\x1f" + participant.loc[valid]
    return result


def match_mri_to_clinical(
    clinical: pd.DataFrame, mri: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match each MRI to a same-person/site clinical visit occurring on or after the MRI.

    A match is eligible when 0 <= clinical visit date - MRI date <= configured limit. This is a
    deliberately conservative temporal rule to avoid using an MRI acquired after its assessment.
    Each MRI can match at most one clinical visit; multiple MRIs can match the same visit, but the
    cohort builder later retains only the closest MRI for a given clinical visit.
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
    clinical_rows_input, mri_rows_input = len(clin), len(scans)
    invalid_clinical_dates = int(clin["_clinical_date"].isna().sum())
    invalid_mri_dates = int(scans["_mri_date"].isna().sum())
    clin = clin.loc[clin["_clinical_date"].notna()].copy()
    scans = scans.loc[scans["_mri_date"].notna()].copy()
    clinical_valid_date_rows, mri_valid_date_rows = len(clin), len(scans)
    clin["_join_key"] = _key_series(clin, cc["participant_id"], cc["site_id"])
    scans["_join_key"] = _key_series(scans, mc["participant_id"], mc["site_id"])
    invalid_clinical_keys = int(clin["_join_key"].isna().sum())
    invalid_mri_keys = int(scans["_join_key"].isna().sum())
    clin = clin.loc[clin["_join_key"].notna()].copy()
    scans = scans.loc[scans["_join_key"].notna()].copy()

    max_delta = int(config["cohort"]["max_mri_clinical_delta_days"])
    matches: list[dict[str, Any]] = []
    no_candidate = 0
    clin_groups = {
        key: group.sort_values(["_clinical_date"], kind="stable")
        for key, group in clin.groupby("_join_key", sort=False)
    }
    for _, scan in scans.iterrows():
        candidates = clin_groups.get(scan["_join_key"])
        if candidates is None or candidates.empty:
            no_candidate += 1
            continue
        delta = (candidates["_clinical_date"] - scan["_mri_date"]).dt.days
        eligible = delta.loc[delta.between(0, max_delta)]
        if eligible.empty:
            no_candidate += 1
            continue
        min_delta = int(eligible.min())
        chosen_idx = eligible.index[eligible.eq(min_delta)][0]
        visit = clin.loc[chosen_idx]
        row = {column: scan[column] for column in mri.columns}
        row.update({f"clinical_{column}": visit[column] for column in clinical.columns})
        row["mri_date"] = scan["_mri_date"]
        row["visit_date"] = visit["_clinical_date"]
        row["delta_days"] = min_delta
        row["_join_key"] = scan["_join_key"]
        matches.append(row)

    matched = pd.DataFrame(matches)
    if matched.empty:
        raise ValueError(
            "No MRI/clinical matches found under the MRI-on-or-before-assessment rule. "
            "Check date columns, participant/site keys, and max_mri_clinical_delta_days."
        )
    deltas = matched["delta_days"].astype(int)
    audit: dict[str, Any] = {
        "clinical_rows_input": clinical_rows_input,
        "mri_rows_input": mri_rows_input,
        "clinical_rows_with_valid_date": int(clinical_valid_date_rows),
        "mri_rows_with_valid_date": int(mri_valid_date_rows),
        "invalid_clinical_dates": invalid_clinical_dates,
        "invalid_mri_dates": invalid_mri_dates,
        "clinical_rows_missing_join_key": invalid_clinical_keys,
        "mri_rows_missing_join_key": invalid_mri_keys,
        "mri_rows_without_eligible_clinical_match": no_candidate,
        "matched_mri_rows": len(matched),
        "matched_participants": int(matched["_join_key"].nunique()),
        "median_clinical_minus_mri_days": float(deltas.median()),
        "p95_clinical_minus_mri_days": float(deltas.quantile(0.95)),
        "max_clinical_minus_mri_days": int(deltas.max()),
        "future_mri_matches": int((deltas < 0).sum()),
        "mri_match_rule": f"MRI date <= clinical date <= MRI date + {max_delta} days",
    }
    return matched, audit


def _clean_feature_values(frame: pd.DataFrame, config: dict[str, Any], prefix: str = "clinical_") -> pd.DataFrame:
    """Coerce configured clinical features to numeric and apply only explicitly configured missing codes."""
    out = frame.copy()
    missing_codes = config.get("feature_missing_codes", {})
    for feature in config["clinical"]["features"]:
        column = f"{prefix}{feature}"
        if column not in out:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        codes = missing_codes.get(feature, [])
        if codes:
            values = values.mask(values.isin([float(code) for code in codes]))
        out[column] = values
        out[f"{column}_missing"] = values.isna().astype("int8")
    return out


def _clinical_timeline(clinical: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Prepare the complete clinical timeline independently of MRI availability."""
    cc = config["clinical"]
    required = [cc[k] for k in ("participant_id", "site_id", "visit_date", "diagnosis", "alzheimers_etiology")]
    _require_columns(clinical, required + list(cc["features"]), "Clinical timeline")
    timeline = clinical.copy().reset_index(drop=True)
    timeline["_date"] = _as_date(timeline[cc["visit_date"]])
    timeline["_join_key"] = _key_series(timeline, cc["participant_id"], cc["site_id"])
    timeline["_diagnosis"] = pd.to_numeric(timeline[cc["diagnosis"]], errors="coerce")
    timeline["_etiology"] = pd.to_numeric(timeline[cc["alzheimers_etiology"]], errors="coerce")
    timeline = timeline.loc[timeline["_date"].notna() & timeline["_join_key"].notna()].copy()
    timeline = timeline.sort_values(["_join_key", "_date"], kind="stable")
    return timeline


def build_cohort(
    matched: pd.DataFrame, clinical: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build one MCI anchor and a short matched sequence per participant.

    The full clinical timeline determines the outcome and follow-up. Only MRI-matched visits are
    eligible model inputs. A positive is an AD-dementia diagnosis in (anchor, anchor + horizon]; a
    negative requires complete observed follow-up through the horizon and no dementia diagnosis
    of another/unknown etiology in that window. Competing/ambiguous dementia events are excluded.
    """
    cc, cohort, codes = config["clinical"], config["cohort"], config["diagnosis_codes"]
    pid_col = f"clinical_{cc['participant_id']}"
    required = [pid_col, f"clinical_{cc['site_id']}", f"clinical_{cc['diagnosis']}",
                f"clinical_{cc['alzheimers_etiology']}", "visit_date", "mri_date", "delta_days"]
    required += [f"clinical_{feature}" for feature in cc["features"]]
    _require_columns(matched, required, "Matched data")

    data = matched.copy().reset_index(drop=True)
    data["visit_date"] = _as_date(data["visit_date"])
    data["mri_date"] = _as_date(data["mri_date"])
    data = data.loc[data["visit_date"].notna() & data["mri_date"].notna()].copy()
    data["_join_key"] = _key_series(data, pid_col, f"clinical_{cc['site_id']}")
    data = data.loc[data["_join_key"].notna()].copy()
    data = _clean_feature_values(data, config)
    data["_diagnosis"] = pd.to_numeric(data[f"clinical_{cc['diagnosis']}"], errors="coerce")
    data["_etiology"] = pd.to_numeric(data[f"clinical_{cc['alzheimers_etiology']}"], errors="coerce")
    data["delta_days"] = pd.to_numeric(data["delta_days"], errors="coerce")
    data = data.sort_values(["_join_key", "visit_date", "delta_days", "mri_date"], kind="stable")
    # At most one scan per clinical visit: keep the scan closest to that visit, with stable tie-breaking.
    data = data.drop_duplicates(subset=["_join_key", "visit_date"], keep="first")

    timeline = _clinical_timeline(clinical, config)
    window = int(cohort["progression_window_days"])
    min_visits = int(cohort["minimum_sequence_visits"])
    max_visits = int(cohort["maximum_sequence_visits"])
    mci_code, dementia_code, ad_code = codes["mci"], codes["dementia"], codes["alzheimers_etiology"]
    manifest_rows: list[dict[str, Any]] = []
    sequence_rows: list[pd.DataFrame] = []
    exclusion_counts = {
        "participants_without_matched_visits": 0,
        "participants_without_mci_anchor": 0,
        "participants_without_sufficient_matched_history": 0,
        "participants_with_competing_or_ambiguous_dementia": 0,
        "participants_with_insufficient_followup": 0,
        "participants_with_no_valid_anchor": 0,
    }

    matched_groups = {key: group for key, group in data.groupby("_join_key", sort=False)}
    timeline_groups = {key: group for key, group in timeline.groupby("_join_key", sort=False)}
    all_keys = set(timeline_groups)
    exclusion_counts["participants_without_matched_visits"] = len(all_keys - set(matched_groups))

    for join_key, timeline_group in timeline_groups.items():
        matched_group = matched_groups.get(join_key)
        if matched_group is None or matched_group.empty:
            continue
        timeline_group = timeline_group.sort_values("_date", kind="stable")
        matched_group = matched_group.sort_values("visit_date", kind="stable")
        anchors = matched_group.loc[matched_group["_diagnosis"].eq(mci_code)]
        if anchors.empty:
            exclusion_counts["participants_without_mci_anchor"] += 1
            continue

        eligible: list[tuple[pd.Series, int, int, int]] = []
        saw_history_failure = False
        saw_competing = False
        saw_followup_failure = False
        for _, anchor in anchors.iterrows():
            anchor_date = anchor["visit_date"]
            past_matched = matched_group.loc[matched_group["visit_date"].le(anchor_date)]
            if len(past_matched) < min_visits:
                saw_history_failure = True
                continue

            # An MCI anchor following an already-recorded dementia diagnosis is inconsistent for this task.
            previous_or_same = timeline_group.loc[timeline_group["_date"].le(anchor_date)]
            prior_dementia = previous_or_same["_diagnosis"].eq(dementia_code)
            if prior_dementia.any():
                saw_competing = True
                continue

            future = timeline_group.loc[
                timeline_group["_date"].gt(anchor_date)
                & timeline_group["_date"].le(anchor_date + pd.Timedelta(days=window))
            ].copy()
            future["_days_from_anchor"] = (future["_date"] - anchor_date).dt.days
            dementia_visits = future.loc[future["_diagnosis"].eq(dementia_code)]
            ad_events = dementia_visits.loc[dementia_visits["_etiology"].eq(ad_code)]
            non_ad_or_unknown = dementia_visits.loc[~dementia_visits["_etiology"].eq(ad_code)]

            if not ad_events.empty:
                label = 1
                days_to_event = int((ad_events["_date"] - anchor_date).dt.days.min())
            elif not non_ad_or_unknown.empty:
                # Do not call a competing or unknown-etiology dementia outcome a non-progression.
                saw_competing = True
                continue
            else:
                latest_followup = timeline_group["_date"].max()
                followup_days = int((latest_followup - anchor_date).days)
                if followup_days < window:
                    saw_followup_failure = True
                    continue
                label = 0
                days_to_event = -1

            eligible.append((anchor, label, days_to_event, int((timeline_group["_date"].max() - anchor_date).days)))

        if not eligible:
            if saw_competing:
                exclusion_counts["participants_with_competing_or_ambiguous_dementia"] += 1
            elif saw_followup_failure:
                exclusion_counts["participants_with_insufficient_followup"] += 1
            elif saw_history_failure:
                exclusion_counts["participants_without_sufficient_matched_history"] += 1
            else:
                exclusion_counts["participants_with_no_valid_anchor"] += 1
            continue

        # Latest eligible MCI anchor ensures one sample per participant.
        anchor, label, days_to_event, followup_days = max(eligible, key=lambda item: item[0]["visit_date"])
        anchor_date = anchor["visit_date"]
        sequence = matched_group.loc[matched_group["visit_date"].le(anchor_date)].tail(max_visits).copy()
        if len(sequence) < min_visits:
            exclusion_counts["participants_without_sufficient_matched_history"] += 1
            continue
        sequence = sequence.sort_values("visit_date", kind="stable").reset_index(drop=True)
        sequence["label"] = int(label)
        sequence["anchor_date"] = anchor_date
        sequence["days_from_anchor"] = (sequence["visit_date"] - anchor_date).dt.days.astype(int)
        intervals = sequence["visit_date"].diff().dt.days
        sequence["delta_time_days"] = intervals.fillna(0).astype(int)
        sequence["is_anchor"] = sequence["visit_date"].eq(anchor_date).astype("int8")
        sequence["sequence_length"] = len(sequence)
        sequence["_join_key"] = join_key
        sequence_rows.append(sequence)
        manifest_rows.append({
            "participant_id": str(join_key),
            "_join_key": join_key,
            "anchor_date": anchor_date,
            "label": int(label),
            "days_to_ad": int(days_to_event),
            "followup_days": int(followup_days),
            "sequence_length": len(sequence),
            "prior_matched_count": int((matched_group["visit_date"] < anchor_date).sum()),
        })

    manifest = pd.DataFrame(manifest_rows)
    sequences = pd.concat(sequence_rows, ignore_index=True) if sequence_rows else pd.DataFrame()
    if manifest.empty:
        raise ValueError(
            "No eligible participants after outcome, follow-up, and sequence criteria. "
            f"Exclusion audit: {exclusion_counts}"
        )
    if manifest["_join_key"].duplicated().any():
        raise AssertionError("Cohort manifest contains duplicate participant/site keys.")
    if not sequences.empty:
        anchor_counts = sequences.groupby("_join_key")["is_anchor"].sum()
        if not anchor_counts.eq(1).all():
            raise AssertionError("Every retained sequence must contain exactly one anchor visit.")
        if (sequences["delta_time_days"] < 0).any():
            raise AssertionError("Elapsed intervals between consecutive visits cannot be negative.")

    cohort_audit: dict[str, Any] = {
        "clinical_timeline_participants": int(timeline["_join_key"].nunique()),
        "participants_with_matched_visits": int(data["_join_key"].nunique()),
        "eligible_participants": len(manifest),
        "label_counts": {str(int(k)): int(v) for k, v in manifest["label"].value_counts().sort_index().items()},
        "sequence_length_counts": {
            str(int(k)): int(v) for k, v in manifest["sequence_length"].value_counts().sort_index().items()
        },
        "exclusions": exclusion_counts,
        "outcome_definition": {
            "positive": f"NACCUDSD={dementia_code} and NACCALZD={ad_code} after anchor and within {window} days",
            "negative": f"no dementia event in the window and at least {window} days of full clinical follow-up",
            "competing_events": "dementia with non-AD or unknown etiology is excluded",
            "anchor": f"matched MCI visit (diagnosis code {mci_code}) with sufficient earlier matched history",
        },
    }
    return manifest, sequences, cohort_audit


def make_patient_splits(manifest: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[str]]:
    """Create stratified patient-level splits; participants never cross split boundaries."""
    _require_columns(manifest, ["participant_id", "label"], "Cohort manifest")
    if manifest["participant_id"].isna().any() or manifest["participant_id"].duplicated().any():
        raise ValueError("Manifest must contain one non-null row per participant before splitting.")
    labels = pd.to_numeric(manifest["label"], errors="raise").astype(int)
    if not set(labels.unique()).issubset({0, 1}) or labels.nunique() != 2:
        raise ValueError("Both binary classes (0 and 1) must be present for stratified splitting.")
    ids = manifest["participant_id"].astype(str)
    seed = int(config["cohort"]["random_seed"])
    train_frac = float(config["cohort"]["train_fraction"])
    val_frac = float(config["cohort"]["validation_fraction"])
    test_frac = float(config["cohort"]["test_fraction"])
    if labels.value_counts().min() < 3:
        raise ValueError("Need at least 3 participants in each class for a three-way stratified split.")
    try:
        train_ids, rest_ids, _, rest_y = train_test_split(
            ids, labels, train_size=train_frac, random_state=seed, stratify=labels
        )
        relative_test = test_frac / (val_frac + test_frac)
        val_ids, test_ids = train_test_split(
            rest_ids, test_size=relative_test, random_state=seed, stratify=rest_y
        )
    except ValueError as exc:
        raise ValueError(
            "The cohort is too small or class-imbalanced for the configured stratified split fractions. "
            "Increase cohort size or adjust the fractions."
        ) from exc

    splits = {
        "train": sorted(train_ids.astype(str).tolist()),
        "validation": sorted(val_ids.astype(str).tolist()),
        "test": sorted(test_ids.astype(str).tolist()),
    }
    sets = [set(values) for values in splits.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Patient leakage detected between splits.")
    if set.union(*sets) != set(ids):
        raise AssertionError("Split membership does not cover the complete cohort.")
    return splits


def _subject_key(join_key: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), join_key.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"subj_{digest[:24]}"


def _safe_output_frames(
    manifest: pd.DataFrame,
    sequences: pd.DataFrame,
    splits: dict[str, list[str]],
    config: dict[str, Any],
    secret: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[str, str]]:
    """Pseudonymize IDs and remove source participant/site identifier columns before writing."""
    cc, mc = config["clinical"], config["mri"]
    key_to_subject = {
        str(join_key): _subject_key(str(join_key), secret)
        for join_key in manifest["_join_key"].astype(str).unique()
    }
    manifest_out = manifest.copy()
    manifest_out["subject_key"] = manifest_out["_join_key"].astype(str).map(key_to_subject)
    manifest_out = manifest_out.drop(columns=["participant_id", "_join_key"], errors="ignore")
    manifest_out = manifest_out[["subject_key"] + [c for c in manifest_out.columns if c != "subject_key"]]

    seq_out = sequences.copy()
    seq_out["subject_key"] = seq_out["_join_key"].astype(str).map(key_to_subject)
    protected_names = {
        cc["participant_id"], cc["site_id"], mc["participant_id"], mc["site_id"],
        f"clinical_{cc['participant_id']}", f"clinical_{cc['site_id']}",
        f"mri_{mc['participant_id']}", f"mri_{mc['site_id']}",
        "_join_key", "_diagnosis", "_etiology",
    }
    # Also remove raw MRI/clinical key fields in case the input has duplicate aliases.
    protected_names |= {name for name in seq_out.columns if name in protected_names}
    seq_out = seq_out.drop(columns=list(protected_names), errors="ignore")
    seq_out = seq_out[["subject_key"] + [c for c in seq_out.columns if c != "subject_key"]]

    pseudonym_splits = {
        split_name: sorted(key_to_subject[participant] for participant in participant_ids)
        for split_name, participant_ids in splits.items()
    }
    return manifest_out, seq_out, pseudonym_splits, key_to_subject


def _fit_training_preprocessor(
    sequences: pd.DataFrame,
    manifest: pd.DataFrame,
    splits: dict[str, list[str]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create model-ready clinical columns using training-participant statistics only.

    Continuous features are median-imputed and standardized from training visits only. NACCSEX is
    one-hot encoded using categories observed in training visits. Missingness flags are preserved.
    This is a deterministic baseline preprocessing contract, not a substitute for validating the
    NACC dictionary's missing-value conventions.
    """
    cc = config["clinical"]
    features = list(cc["features"])
    sex_feature = config.get("categorical_features", ["NACCSEX"])
    categorical = [feature for feature in sex_feature if feature in features]
    continuous = [feature for feature in features if feature not in categorical]
    train_ids = set(splits["train"])
    train_keys = set(manifest.loc[manifest["participant_id"].astype(str).isin(train_ids), "_join_key"].astype(str))
    train = sequences.loc[sequences["_join_key"].astype(str).isin(train_keys)]
    if train.empty:
        raise ValueError("No training-participant visits available to fit clinical preprocessing.")

    out = sequences.copy()
    preprocessing: dict[str, Any] = {"fit_scope": "training participants only", "continuous": {}, "categorical": {}}
    for feature in continuous:
        col = f"clinical_{feature}"
        values = pd.to_numeric(out[col], errors="coerce")
        train_values = pd.to_numeric(train[col], errors="coerce")
        median = float(train_values.median()) if train_values.notna().any() else 0.0
        imputed_train = train_values.fillna(median)
        mean = float(imputed_train.mean())
        std = float(imputed_train.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        imputed_all = values.fillna(median)
        out[f"{feature}__scaled"] = (imputed_all - mean) / std
        out[f"{feature}__missing"] = values.isna().astype("int8")
        preprocessing["continuous"][feature] = {"median": median, "mean": mean, "std": std}

    for feature in categorical:
        col = f"clinical_{feature}"
        values = pd.to_numeric(out[col], errors="coerce")
        train_values = pd.to_numeric(train[col], errors="coerce")
        categories = sorted(float(value) for value in train_values.dropna().unique())
        for category in categories:
            suffix = str(int(category)) if category.is_integer() else str(category).replace(".", "_")
            out[f"{feature}__category_{suffix}"] = values.eq(category).astype("int8")
        out[f"{feature}__missing"] = values.isna().astype("int8")
        preprocessing["categorical"][feature] = {"categories": categories, "unknown_category": "all-zero one-hot"}

    model_columns = ["_join_key", "subject_key", "visit_date", "mri_date", "delta_days", "delta_time_days",
                     "days_from_anchor", "is_anchor", "sequence_length", "label"]
    model_columns += [
        column
        for column in out.columns
        if column.endswith(("__scaled", "__missing")) or "__category_" in column
    ]
    # subject_key is added only after split-safe mapping by write_outputs; this function's intermediate
    # frame is keyed by the internal join key and is never written directly.
    model_columns = [column for column in model_columns if column in out.columns]
    model_ready = out[model_columns].copy()
    return model_ready, preprocessing


def _write_audit_figures(audit: dict[str, Any], output_dir: Path) -> None:
    """Write simple aggregate audit figures; no participant identifiers appear in these plots."""
    # Import plotting support only when figures are requested, so core cohort tests do not
    # fail during module import if optional plotting dependencies are not installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    label_counts = audit.get("label_counts", {})
    if label_counts:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(list(label_counts), list(label_counts.values()))
        ax.set(title="Cohort label counts", xlabel="Label", ylabel="Participants")
        fig.tight_layout()
        fig.savefig(output_dir / "audit_label_counts.png", dpi=150)
        plt.close(fig)

    split_counts = audit.get("split_counts", {})
    if split_counts:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(list(split_counts), list(split_counts.values()))
        ax.set(title="Patient split counts", xlabel="Split", ylabel="Participants")
        fig.tight_layout()
        fig.savefig(output_dir / "audit_split_counts.png", dpi=150)
        plt.close(fig)

    missingness = audit.get("clinical_feature_missing_rates", {})
    if missingness:
        fig, ax = plt.subplots(figsize=(8, 4))
        names, rates = list(missingness), list(missingness.values())
        ax.bar(names, rates)
        ax.set(title="Clinical feature missingness", ylabel="Fraction missing", ylim=(0, 1))
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / "audit_feature_missingness.png", dpi=150)
        plt.close(fig)


def write_outputs(
    manifest: pd.DataFrame,
    sequences: pd.DataFrame,
    matched_audit: dict[str, Any],
    cohort_audit: dict[str, Any],
    splits: dict[str, list[str]],
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    identifier_secret: str | None = None,
) -> dict[str, Any]:
    """Write local Parquet/JSON/PNG outputs with pseudonymous identifiers only.

    Set WORKSTREAM1_ID_SECRET in the environment to a high-entropy secret. The same secret is
    needed to reproduce subject keys across runs. The secret is never written to disk.
    """
    secret = identifier_secret or os.environ.get("WORKSTREAM1_ID_SECRET")
    if not secret or len(secret) < 24:
        raise ValueError(
            "Set WORKSTREAM1_ID_SECRET to a stable secret of at least 24 characters before writing outputs."
        )
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest_out, sequences_out, splits_out, key_to_subject = _safe_output_frames(
        manifest, sequences, splits, config, secret
    )
    model_ready, preprocessing = _fit_training_preprocessor(sequences, manifest, splits, config)
    model_ready["subject_key"] = model_ready["_join_key"].astype(str).map(key_to_subject)
    model_ready = model_ready.drop(columns=["_join_key"], errors="ignore")
    model_ready = model_ready[["subject_key"] + [c for c in model_ready.columns if c != "subject_key"]]

    # Parquet is the canonical tabular format. pyarrow is a required project dependency.
    manifest_out.to_parquet(out / "cohort_manifest.parquet", index=False)
    sequences_out.to_parquet(out / "visits_clinical.parquet", index=False)
    model_ready.to_parquet(out / "visits_clinical_model_ready.parquet", index=False)
    with (out / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(splits_out, handle, indent=2)
    with (out / "preprocessing.json").open("w", encoding="utf-8") as handle:
        json.dump(preprocessing, handle, indent=2)

    split_map = {participant: split_name for split_name, participants in splits.items() for participant in participants}
    labeled = manifest.copy()
    labeled["split"] = labeled["participant_id"].astype(str).map(split_map)
    features = [f"clinical_{feature}" for feature in config["clinical"]["features"]]
    missing_rates = {
        column.removeprefix("clinical_"): float(sequences[column].isna().mean())
        for column in features if column in sequences.columns
    }
    actual_leakage = 0
    split_sets = {name: set(values) for name, values in splits.items()}
    names = list(split_sets)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            actual_leakage += len(split_sets[first] & split_sets[second])
    audit: dict[str, Any] = {
        "source_freeze": config["source_freeze"],
        "matched_data": matched_audit,
        "cohort": cohort_audit,
        "eligible_participants": len(manifest),
        "label_counts": {str(int(k)): int(v) for k, v in manifest["label"].value_counts().sort_index().items()},
        "sequence_length_counts": {
            str(int(k)): int(v) for k, v in manifest["sequence_length"].value_counts().sort_index().items()
        },
        "split_counts": {name: len(values) for name, values in splits.items()},
        "split_label_counts": {
            name: {str(int(k)): int(v) for k, v in labeled.loc[labeled["split"].eq(name), "label"].value_counts().sort_index().items()}
            for name in splits
        },
        "patient_leakage": int(actual_leakage),
        "clinical_feature_missing_rates": missing_rates,
        "identifier_policy": "HMAC-SHA256 pseudonymous subject_key; raw participant/site IDs omitted from outputs",
        "preprocessing": "Median imputation and standardization fit on training participants only; NACCSEX one-hot encoded",
        "data_protection_warning": "All participant-level outputs are restricted; never commit them to Git.",
    }
    with (out / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=str)
    _write_audit_figures(audit, out)
    return audit
