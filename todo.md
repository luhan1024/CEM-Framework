# CEM-Framework repository migration log

This file tracks the repository relocation and package-layout work separately
from scientific manuscript and legacy Case 1--6 outputs.

## A. Stage-2 estimator release gate

- **Completed and pushed.** Commit: `f488d844c8c7e7446bd78e1e7d8d06d1812fdc7c`
  (`Add narrow 1D equilibrated-flux estimator`), based on
  `08354e2d3b5400c842dbcd662abe5081be8853b6`.
- Local evidence on TensorFlow Conda Python 3.11.13: `compileall` passed,
  unittest passed 62/62, all three examples passed, wheel build passed, and
  `git diff --check` passed. The wheel included
  `cem_framework/estimators_1d.py`.
- GitHub Actions run `42991047091` completed successfully for the exact commit
  SHA. All six Ubuntu/Windows × Python 3.10/3.11/3.12 jobs are green.

## B. Relocate the repository after A is green

- Original Git root: `Code/cem_framework/` (nested under the legacy `Code/`
  tree; `Code/` has a separate Git root).
- Chosen destination: project-root sibling `CEM-Framework/`.
- **Completed:** created `CEM-Framework/` with a history-preserving local clone
  at the exact A commit. After the new layout passed local validation, moved
  the unchanged legacy checkout to `CEM-Framework-legacy-checkout/`; it remains
  a complete Git checkout and is no longer nested under `Code/`.
- Do not migrate or modify legacy Case scripts, reviewer assets, manuscript
  sources, figures, data, or generated paper outputs.

## C. Standardize package layout in the relocated repository

Path mapping:

| Current repository path | Destination repository path |
| --- | --- |
| flat package modules (`__init__.py`, `solver.py`, etc.) | `src/cem_framework/` |
| `tests/` | `tests/` |
| `examples/` | `examples/` |
| `docs/` | `docs/` |
| `.github/workflows/`, `pyproject.toml`, `requirements.txt`, `README.md`, `LICENSE` | repository root, same relative paths |

**Completed locally:** moved the tracked modules using Git, configured standard
Setuptools `src` discovery, updated README commands and direct examples, and
included `requirements.txt` in the Actions pip-cache key. Public API names and
default behavior are unchanged.

## D. Build hygiene and preservation

- Updated `.gitignore` to cover build/dist/egg-info, `__pycache__`, virtual
  environments, pytest/tox caches, wheelhouses, temporary build directories,
  and built wheels.
- The `Code/` Case scripts and data were not moved or deleted. No Case scripts
  referenced the old `Code/cem_framework` package path.
- Keep workspace temporary PDF/PNG/LaTeX outputs and reviewer assets outside
  the new repository; do not perform broad workspace cleanup.

## E. Relocated-repository validation and release

- **Local validation passed** on the TensorFlow Conda Python 3.11.13:
  editable install resolves `cem_framework` from `CEM-Framework/src`; compileall
  passed; unittest passed 62/62; all three examples passed; wheel built and
  contains `cem_framework/estimators_1d.py`; `git diff --check` passed.
- Wheel: `cem_framework-0.1.0-py3-none-any.whl`, SHA-256
  `91873d2f323cc554810844dfd469fd4ec0340a093be439385e094fc75331bc29`.
- Migration commit `3d126a2a7bf6368d6694d2c36e774292c78a4ed6` was pushed to
  `main`. Actions run `42992713311` completed successfully for that exact SHA;
  Ubuntu/Windows × Python 3.10/3.11/3.12 are all green.
