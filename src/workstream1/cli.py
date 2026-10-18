"""Command-line entry point for the local Workstream 1 cohort pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .pipeline import (
    build_cohort,
    load_config,
    make_patient_splits,
    match_mri_to_clinical,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a restricted longitudinal clinical/MRI cohort from local NACC CSV files."
    )
    parser.add_argument("--clinical", required=True, help="Local path to the authorized clinical CSV")
    parser.add_argument("--mri", required=True, help="Local path to the authorized MRI CSV")
    parser.add_argument("--config", default="configs/nacc_columns.json", help="Pipeline JSON config")
    parser.add_argument("--out", default="data/processed", help="Local restricted output directory")
    args = parser.parse_args()

    if not os.environ.get("WORKSTREAM1_ID_SECRET") or len(os.environ["WORKSTREAM1_ID_SECRET"]) < 24:
        parser.error(
            "Set WORKSTREAM1_ID_SECRET to a stable secret of at least 24 characters before running. "
            "Do not store this secret in the repository or config file."
        )

    config = load_config(args.config)
    clinical = pd.read_csv(args.clinical, low_memory=False)
    mri = pd.read_csv(args.mri, low_memory=False)
    matched, match_audit = match_mri_to_clinical(clinical, mri, config)
    manifest, sequences, cohort_audit = build_cohort(matched, clinical, config)
    splits = make_patient_splits(manifest, config)
    audit = write_outputs(manifest, sequences, match_audit, cohort_audit, splits, config, args.out)

    print("Workstream 1 completed.")
    print(f"Source freeze: {config['source_freeze']}")
    print(f"Eligible participants: {audit['eligible_participants']}")
    print(f"Label counts: {audit['label_counts']}")
    print(f"Split counts: {audit['split_counts']}")
    print(f"Patient leakage: {audit['patient_leakage']}")
    print(f"Outputs written locally to: {Path(args.out).resolve()}")
    print("Do not commit participant-level outputs or the identifier secret to GitHub.")


if __name__ == "__main__":
    main()
