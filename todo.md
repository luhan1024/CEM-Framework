# CEM-Framework repository migration log

This file tracks the repository relocation and package-layout work separately
from scientific manuscript and legacy Case 1--6 outputs.

## A. Stage-2 estimator release gate

- **Completed and pushed.** Commit: `f488d844c8c7e7446bd78e1e7d8d06d1812fdc7c`
  (`Add narrow 1D equilibrated-flux estimator`), based on
  `08354e2d3b5400c842dbcd662abe5081be8853b6`.
- Local evidence on TensorFlow Conda Python 3.11.13: `compileall` passed,
  unittest passed 62/62, all three examples passed, wheel build passed, and
  `git diff --check` passed. The wheel contained the estimator module in the
  then-current package layout; this is historical Stage-A evidence only.
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

## C. Standardize the package layout in the relocated repository

Path mapping:

| Current repository path | Destination repository path |
| --- | --- |
| flat modules (`solver.py`, `estimators_1d.py`) | `src/` |
| state, ledger, and cost modules | `src/core/` |
| problem definitions and operators | `src/problems/` |
| learners, candidate pools, and serialization | `src/enrichment/` |
| scorers and acceptance policies | `src/strategies/` |
| `tests/` | `tests/` |
| `examples/` | `examples/` |
| `docs/` | `docs/` |
| `.github/workflows/`, `pyproject.toml`, `requirements.txt`, `README.md`, `LICENSE` | repository root, same relative paths |

The user selected direct imports from these responsibility packages/modules,
with no umbrella namespace. The short names are generic and can collide with
unrelated imports; README and API docs state this risk.

## D. Build hygiene and preservation

- Updated `.gitignore` to cover build/dist/egg-info, `__pycache__`, virtual
  environments, pytest/tox caches, wheelhouses, temporary build directories,
  and built wheels.
- The `Code/` Case scripts and data were not moved or deleted. No Case scripts
  referenced the old `Code/cem_framework` package path.
- Keep workspace temporary PDF/PNG/LaTeX outputs and reviewer assets outside
  the new repository; do not perform broad workspace cleanup.

## E. Relocated-repository validation and release (before responsibility regrouping)

- Historical pre-regrouping validation passed on the TensorFlow Conda Python
  3.11.13: editable install, compileall, 62/62 tests, three examples, wheel
  build/content check, and `git diff --check`.
- Historical distribution artifact: `cem_framework-0.1.0-py3-none-any.whl`, SHA-256
  `91873d2f323cc554810844dfd469fd4ec0340a093be439385e094fc75331bc29`.
- These results validate the previous package layout only; they do not describe
  the current direct top-level responsibility packages or their wheel contents.
- Migration commit `3d126a2a7bf6368d6694d2c36e774292c78a4ed6` was pushed to
  `main`. Actions run `42992713311` completed successfully for that exact SHA;
  Ubuntu/Windows × Python 3.10/3.11/3.12 are all green.

## F. Responsibility-package regrouping (current architecture-only slice)

- Reorganized the pending worktree to the user-selected direct layout:
  `src/core/`, `src/problems/`, `src/enrichment/`, `src/strategies/`, plus
  `src/solver.py` and `src/estimators_1d.py`. No umbrella package is retained.
- Subpackages provide narrow `__init__.py` exports; the public examples import
  from the selected top-level modules/packages. `pyproject.toml` auto-discovers
  packages under `src/` and explicitly lists `solver` and `estimators_1d` as
  root modules.
- TensorFlow Conda Python 3.11.13 editable install resolves the four packages
  and two modules from the expected `src/` paths. `compileall` passed;
  unittest passed 66/66 (including four architecture tests); all three examples
  passed; `git diff --check` passed.
- Wheel content check found all 16 expected Python module paths, no duplicate
  archive paths, and no umbrella package. Wheel artifact
  `cem_framework-0.1.0-py3-none-any.whl` SHA-256:
  `c2bbad5bbf114d66e122dbc0b311beaaceec3f0f9e5940983c6eef2ad9c2b55a`.
- This is architecture/import organization only. It is not a P1 certificate,
  scientific validation, PDE experiment, or manuscript evidence.
- Local validation and wheel verification passed. Architecture commit
  `9e5b80a20f2694b209e434e2575f2fcd394ef734` was pushed; Actions run
  `42996391679` succeeded for that exact SHA across Ubuntu/Windows × Python
  3.10/3.11/3.12. This remains architecture-only software evidence.

## G. Fixed-discrete Ritz P1 verifier (narrow prototype; review pending)

- Added an independent verifier under `src/problems/`; it consumes a
  `DiscreteRitzProblem` plus finite float32/float64 state, direction, and
  enclosure values, interpreted as exact dyadic rationals. It exactly checks
  symmetry, SPD, directional curvature, declared `c/q` enclosures, and forms
  the rational step/lower bound/trial. It does not call or modify `CEMSolver`.
- The public entry point consumes `DiscreteRitzProblem` and reads its actual
  `feasible` property. The first version is restricted to
  `problem.feasible is None`; callbacks are rejected as unsupported.
  `feasibility_basis` records `unconstrained_domain` only in that case. Exact
  factor work is capped at dimension 32, 6,000 LDL summation products, and 16,384 bits per
  Fraction numerator/denominator. Oversized inputs fail closed as
  `unsupported_resource_limit` before input hashing/factorization.
- Added an input-bound JSON record plus `verify_discrete_ritz_certificate`;
  serialized `status` is not proof. Verification recomputes fields/hash and
  independently evaluates the exact objective difference on the exact trial.
- TensorFlow Conda Python 3.11.13: 76/76 unittests passed; `compileall`, all
  three examples, wheel build/content/export check, and `git diff --check`
  passed. The Conda launcher itself failed in this shell, so the environment's
  `python.exe` was invoked directly. Wheel SHA-256:
  `f28759c0a635bf73a0009e728c329fe23d4f62fe8610a447ac6210d85111c8f4`.
- Evidence boundary: this is software verification for a fixed finite
  dyadic SPD quadratic and caller-declared enclosures. It does not verify
  enclosure provenance, continuous PDE integrals/errors, greedy selection,
  convergence, machine-certified proof status, or novelty. No manuscript,
  Case data, figures, or `requirements.txt` were changed. Exact-SHA CI run
  `43000717388` passed all six Ubuntu/Windows × Python 3.10/3.11/3.12 jobs for
  implementation commit `54f2e023a9150f7ecb8b3ae6558cd0c647eba39a`;
  independent scientific review remains pending.

## H. Bounded analytic P1 fixture (general enclosure provenance remains open)

- Added one deterministic test fixture for weak 1D Poisson on `(0,1)` with
  homogeneous Dirichlet data, `f=1`, and `psi=x(1-x)`. It keeps the continuous
  analytic values `c=1/6`, `q=1/3` distinct from the exact dyadic values of the
  float64 `DiscreteRitzProblem` load and stiffness entries.
- The fixture checks the analytic integrals with an independent 80-digit
  Decimal Simpson calculation and checks a separate two-subinterval midpoint
  rule using exact rational arithmetic. Its outward-rounded `delta` and
  `epsilon` are derived exactly to enclose both the continuous analytic `c/q`
  and the verifier's dyadic `c/q` around the midpoint centers. Thus they
  include midpoint quadrature discrepancy and float64 coefficient
  representation error. One-ULP-understated radii are tested independently.
- Independent rational oracles compute discrete and continuous energy
  decreases directly for this analytic fixture; both certificate and verifier
  calls are checked for mutations to the supplied arrays, problem data, and
  attached work ledger. No solver update or reusable continuous-PDE energy
  certificate is claimed.
- TensorFlow Conda Python 3.11.13: `compileall`, all 77 unittests, and all
  three deterministic examples passed. This closes only the reviewed test
  fixture. Reliable, reusable provenance for `delta/epsilon` on general
  candidates and quadrature rules remains **OPEN**; P1 is not closed.
