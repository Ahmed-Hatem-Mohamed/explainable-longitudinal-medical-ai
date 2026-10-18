# Data protection notes

- Source NACC files and all participant-level or derived data must remain on approved local storage.
- Do not commit source files, raw identifiers, cohort manifests, split membership, visit sequences, or preprocessing outputs.
- Generated outputs replace raw participant/site identifiers with keyed HMAC-SHA256 `subject_key` values. Pseudonymization does not make these data public or non-sensitive.
- `WORKSTREAM1_ID_SECRET` must be long, stable across runs when stable pseudonyms are needed, and stored outside the repository. Never print it, include it in the config, or commit it.
- The `feature_missing_codes` config is intentionally empty. Populate it only from the authorized dictionary; do not infer sentinel values from observed distributions.
- Review `git status --short`, `git diff --cached --name-only`, and `.gitignore` checks before every commit.
