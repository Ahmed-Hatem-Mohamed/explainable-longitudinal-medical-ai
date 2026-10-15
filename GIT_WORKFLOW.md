# Suggested GitHub workflow

This is a source-code-only package. Integrate it into the existing repository after a maintainer confirms the target branch names and contribution process.

```powershell
git clone https://github.com/Nada-belarbi/explainable-longitudinal-medical-ai.git
cd explainable-longitudinal-medical-ai
git switch develop
git pull origin develop
git switch -c feature/data-cohort
```

Copy the reviewed source package into the repository root, preserving any existing project files unless the maintainer explicitly approves replacing them. Then verify the repository contains no data artifacts:

```powershell
git status --short
git check-ignore -v data/restricted/clinical.csv
git check-ignore -v data/processed/cohort_manifest.csv
```

Run tests and inspect staged paths before committing:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
git add .gitignore README.md pyproject.toml configs src tests docs GIT_WORKFLOW.md
git diff --cached --stat
git diff --cached --name-only
```

If all staged files are source/config/docs/tests only, commit and push the feature branch, then open a pull request to `develop`:

```powershell
git commit -m "Add Workstream 1 cohort pipeline scaffold"
git push -u origin feature/data-cohort
```

Do not use `git add .` before reviewing ignored/untracked files. If `develop` does not exist or branch protections differ, follow the repository maintainer's instructions instead. This guide does not imply that any files have already been pushed.
