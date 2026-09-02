# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- BSD-3-Clause license and expanded package-index metadata.
- Project skeleton: `src/binaria` package layout, test suite layout, and
  community/distribution scaffolding (`CITATION.cff`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `container/Apptainer.def`, `benchmarks/`).
- Dev tooling: `ruff`, `mypy --strict`, `pytest`, `pytest-cov`, `hypothesis`,
  `import-linter`, and `pre-commit`, configured in `pyproject.toml` and wired
  through `uv run` so pre-commit hooks use the same locked versions as CI.
- `ci.yml`: a lint/typecheck job (ruff, mypy --strict, import-linter) and a
  pytest matrix (Ubuntu x Python 3.11/3.12/3.13, plus one macOS and one
  Windows smoke test on the newest version).
- Issue and pull request templates, following conventions from scikit-learn,
  scvi-tools, and scanpy.
- Optional JSON decision trails and per-fit timing for auditable model selection.

### Changed

- Restricted model-selection criteria to the three implemented names instead
  of exposing a criterion object that could not supply custom scoring logic.
