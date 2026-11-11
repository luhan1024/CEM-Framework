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

## I. Polynomial Poisson record parser depth and Unicode boundary

- Added a bounded JSON container-depth scan before decoding the polynomial
  enclosure record (maximum nesting depth 64), with explicit `RecursionError`
  conversion as a decoder/replay backstop. `from_json` reports deeply nested
  input and lone-surrogate text as defined `ValueError`s; the verifier rejects
  a directly constructed deeply nested record with `False`.
- Added regression coverage using 1,500 nested arrays through both public
  parser/verifier paths, lone-surrogate input, and directly constructed
  records with `None`/integer payloads. Existing duplicate-key, noncanonical
  JSON, and record-size rejection tests remain in place.
- TensorFlow Conda Python 3.11.13: focused polynomial-family tests passed
  10/10; full unittest discovery passed 87/87; `compileall`, all three
  deterministic examples, and `git diff --check` passed for this worktree.
- Scope remains the bounded analytic polynomial Poisson family only. This
  parser hardening does not establish general enclosure provenance or close
  P1/Stage 2; those remain **OPEN**. No manuscript, figures, or data were
  changed. This reviewer-requested worktree update was not committed or pushed.

## J. Stage-2 literature-to-API gap audit

- Added [`docs/stage2-literature-gap-audit.md`](docs/stage2-literature-gap-audit.md):
  page-located matrix for all 17 PDFs (755 pages) and cross-check of all 4
  text companions present (the request expected 3). It maps each source to K0/K1/P1/P2/P3/P4/L1,
  states assumptions and the exact current API gap, and marks unresolved
  theorem-transfer/metadata/novelty questions **OPEN**.
- Reused and linked the existing R1/R2/R3 audits for focused derivations;
  this report avoids restating their long proofs. It identifies the bounded
  exact 1D P1 Poisson cell-integral certificate as one candidate next gate,
  with explicit acceptance evidence, stop conditions, and nonclaims. It is a
  proposal only; no code, manuscript, data, or experiments were changed/run.
- TensorFlow Conda Python 3.11.13 built a local wheel from this worktree, and
  an isolated target install imported `core`, `problems`, `enrichment`, and
  `strategies` successfully. Initial pip temp-path denial was resolved by
  placing TMP/TEMP under the permitted workspace; no external dependencies
  were fetched (`--no-deps`, `--no-build-isolation`, `PIP_NO_INDEX=1`).
- **Stage 2/P1 remains OPEN.** At the end of this audit step, the next item was
  independent review of the exact 1D P1 certificate contract. Section K records
  that later review and bounded software implementation; section L records the
  current unique next gate. The historical finite-pool ratios remain unrelated
  to continuous-dictionary guarantees, and software tests are not scientific
  PDE evidence.

## K. Exact piecewise-linear 1D Poisson certificate contract (design gate)

- Added `docs/p1-piecewise-linear-poisson-certificate-contract.md` as a
  review-only specification: weak-form sign convention and energy identity,
  exact binary64/dyadic input semantics, cellwise `c/q` formulas, signed/zero
  cases, canonical input-bound record/replay, fail-closed statuses and
  proposed resource caps, oracle tests, package integration points, and
  nonclaims.
- Difference audit: discrete Ritz verifies caller-declared enclosures for a
  finite matrix but does not derive continuous source/mesh provenance; the
  polynomial Poisson path covers global polynomial data, not general P1
  knotwise functions; the float64 equilibrated-flux estimator shares the same
  model class but returns a numerical error estimate and does not compute the
  exact directional gain. The proposed feature is **GO only as a bounded exact
  oracle/fixture, NO-GO as a claimed new general estimator or theorem**. If a
  test-only oracle is sufficient, prefer it over a public API.
- **Contract review approved for this narrow implementation.** Added pure
  `certify_p1_poisson_step` / `verify_p1_poisson_step_certificate` in
  `src/problems/p1_poisson_1d.py`, exported through `problems`. It snapshots
  the bounded binary64 input arrays before value validation, hashing, and exact
  integration; derives exact cellwise `c/q`, signed alpha and energy gain;
  independently recomputes direct energy difference; binds/replays canonical
  records; and fails closed on invalid or over-budget inputs. It does not
  connect to the solver/pool. Resource accounting is exact for the current
  path: `36*cells + 11` checked rational operations, with the live counter and
  cap preflight both enforced (147,467 operations at the 4,096-cell cap).
- Added independent Fraction/direct-energy oracle and edge tests, including
  negative/zero `c`, zero direction, signed-zero endpoint semantics and raw-bit
  hash retention, smallest subnormal and largest finite binary64 inputs,
  parser/unhashable-status failures, tampering, snapshot non-mutation, and exact
  max-cell operation count. The largest-finite case independently compares
  `c`, `q`, signed `alpha`, exact gain, and direct energy decrease against the
  test-only `Fraction` oracle; the negative-zero endpoint is legally replayed.
  `P1Poisson1DData` subclasses and ndarray subclasses are rejected.
  The API is documented in `docs/api_reference.md`; the mathematical and
  novelty boundary remains in
  `docs/p1-piecewise-linear-poisson-certificate-contract.md`.
- Final-source validation after the largest-finite and signed-zero regressions
  in TensorFlow Conda Python 3.11.13: full unittest discovery **103/103**; `compileall`
  passed; all three existing
  examples passed; clean wheel build/install plus imports from the installed
  wheel passed for `core`, `problems`, `enrichment`, and `strategies`, including
  the new P1 API. Wheel SHA-256:
  `230d779e3385a5ccc94587527fa600b0b91017ca3acef41608011292978c31ba`.
  `git diff --check` passed. No new PDE benchmark, manuscript, data, or figures
  were changed/run.
- This closes only the approved bounded **software oracle/API gate**. General
  P1 enclosure provenance, PDE solution error, K1/P2/P3/P4, convergence,
  comparative efficiency, and novelty remain **OPEN**. Solver integration is
  explicitly deferred to a separate review.

## L. Stage-2 next gate design and package-layout confirmation

- Audited the current architecture decision against `pyproject.toml`, `src/`,
  docs, tests, examples, and CI: `src/` directly contains `core`, `problems`,
  `enrichment`, `strategies`, plus the root modules `solver.py` and
  `estimators_1d.py`. Public imports and CI tests use those top-level modules;
  no `src/cem_framework/` umbrella package or compatibility namespace exists.
  The project distribution identifier `cem-framework` is distinct from the
  import namespace. The `src/cem_framework.egg-info/` directory is generated,
  ignored build metadata; its `top_level.txt` lists the direct modules only.
  No layout correction is indicated. `docs/api_reference.md` records this
  distinction explicitly.
- Added [`docs/stage2-next-gate.md`](docs/stage2-next-gate.md), a design-only
  comparison of rigorous Gaussian/RBF interval provenance against exact local
  FE/P1 integration. The recommended single next deliverable is a deterministic
  P1 interior-hat weak learner over the fixed finite dictionary, selecting the
  exact best directional gain and returning one certificate-backed Candidate.
  It proposes `delta=epsilon=0` only for the exact bounded P1 model; it does not
  claim solver-update certification, continuous-dictionary coverage, new
  theory, or performance advantage. The document records exact cost accounting,
  independent oracles, stress tests, CandidatePool boundaries, Xu & Xu and
  Mishra–Molinaro/flux-estimator distinctions, expected claims, and No-Go gates.
- **2026-11-11 approval and implementation update:** the reviewer approved the
  bounded proposal-only P1 API. Added `P1Poisson1DRitzProblem` and the
  top-level-module `P1PoissonHatWeakLearner` (`p1_poisson_hat.py`), preserving
  the responsibility-package import boundaries; no `CEMSolver` hookup.
  Metadata cap is checked
  from a canonical pre-scan skeleton; actual/reserved hash and serialization
  bytes are separately recorded, and candidate metadata serialization uses a
  bounded convergence loop whose returned size is verified exactly. API
  reference and this gate note now reflect implementation status. Latest local
  validation: P1 tests 15/15; full suite 118/118; `compileall`; three existing
  examples; `git diff --check`; offline wheel build and isolated install/import
  all passed in TensorFlow Conda Python 3.11.13 (NumPy 2.4.6). Latest wheel
  SHA-256: `d890ebb4e4ee398db163e746ae3b33eeca51e99921dca7364096da84d0de3e21`.
  Independent source review remains pending. No commit or push was made.
- **2026-11-11 final-review correction:** reviewer found the state-fingerprint
  encoding did not distinguish absent outer coefficients from an empty
  float64 vector. Added explicit `none`/`present` domain markers and dtype/
  shape metadata; added a bidirectional fingerprint/change-detection test.
  Re-ran the focused and full suites, examples, compileall, and offline wheel
  validation after the correction; results and replacement wheel hash are
  recorded above.
- **2026-11-11 design clarification revision:** expanded the proposal budget
  schema with exact defaults for omitted keys, strict rejection of malformed
  values and unknown keys, and states that every table maximum is hard: callers
  may omit or lower a cap but may not raise it; insufficient caps reject before
  a nonempty scan rather than scan a dictionary prefix. Defined counts for hat
  evaluations, local-cell visits, adapter cell visits/operator rows, the one
  residual operator-row pass, and their aggregate row bound. For `N>1,K=0`,
  active-ID validation is followed directly by `dictionary_exhausted` with no
  residual recomputation, exact local preprocessing, scan reservation, or
  certificate work; preflight exact rational work and proposal-path work
  counters are zero. Added acceptance checks for `N=1` `empty_dictionary`
  (operator-assembly/residual/candidate/certificate work counts zero; adapter
  data/problem identity hashing at construction remains separately reported)
  and for `tau=1` only
  when `K>0` and maximum gain is positive; empty, exhausted, and all-zero
  cases leave tau/weakness undefined. The former design-only status is
  superseded by the approval and implementation update above. This remains a
  proposal-only API gate: no solver hookup, PDE benchmark, research-plan or
  manuscript edit, data or figure change, commit, or push is in scope.
